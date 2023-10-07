---
layout: post
title:  "Preparing a custom model with Vitis AI"
date:   2023-10-01 12:27:00 -1000
categories: xilinx edge_ai quantization vitis
---

# Preparing a custom model with Vitis AI
The whole point of getting the Vitis AI container downloaded in the last post
was to generate our own model files for the Xilinx DPU. Xilinx has many [premade models available](https://github.com/Xilinx/Vitis-AI/tree/v2.5/model_zoo)
but you'll eventually need to create your own if nothing up there matches your needs.

## FashionMNIST
The model I'm going to be quantizing is a classifier with 2 convs and 2 linear layers trained
on FashionMNIST, a convenient dataset that's readily available through torchvision. To be clear,
it's a minimal example to show the process, and small enough to be trained + quantized on a CPU
in a few minutes. You can find the model training code [here](TODO) in one of my repos.

After training you should see ~90% accuracy, and end up with at least a `fmnist.pt` set of weights.
This is about where you'll be starting if you're following along with a custom model. Up to this point,
there shouldn't be any requirement to use the special Vitis AI docker containers or even match their
PyTorch version.

## Quantization
To create a model for the Xilinx DPU, we need a quantized set of weights. In short, quantization
is the process of converting a model whose weights are in 32bit float precision down to one
with only 8bit integer precision. That's _a lot_ less precision, but also a much smaller range.
After all, int8's can only be -128 to 127! My shallow understanding of the quantization process
is that it will need to find some additional offsets and scale factors for regions of the
network that allow it to be evaluated without saturating the int8's, but to do so it needs to
see some representative examples of what might come in at test time.

Now we're going to need those Xilinx docker images.
On your host/workstation...
```
# Get Vitis-AI v2.5, which I'll be just calling vitis
docker pull xilinx/vitis-ai-cpu:2.5
```
On the Kria...
```
# Get Kria runtime based on Vitis runtime/library v2.5, which I'll call kria-runtime
docker pull xilinx/kria-runtime:2022.1
```
A note here about versions; I began with the Vitis v2.5 host containers, but the outputs from
newer versions like v3.5 appear to be compatible with kria-runtime. If you want CUDA support,
you need to [build it yourself](https://github.com/Xilinx/Vitis-AI/tree/master/docker) though.
I also tried to rebuild v2.5's container with GPU support, but couldn't get the old image to build...
seems like some packages (and even the base image) are no longer available, which is surprising
since that image is not even 2 years old.

### Inspecting
I'll be referring to [my repo]() again here, specifically `fminst_quantize_pt1.py` right now.

The first step, inspecting the model, lets us know if the torch model we want to quantize can
be mapped to the DPU's supported operations, and what outstanding operators will end up mapped
to the CPU runner instead. We need two things to inspect: a dummy tensor matching the shape of
our input data, and the name of the DPU we're going to be compiling for. The dummy input is easy,
but to get the DPU name, you can run `xdputil query` on the Kria (in the container) after loading
the bitstream/app with `xmutil load <appname>` (outside of the container).

{% highlight python %}
target = "DPUCZDX8G_ISA1_B3136" # (or 0x101000016010406)
inspector = Inspector(target)
dummy_input = torch.randn(batchsize, 1, 28, 28).to(device).float()
inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png")
{% endhighlight %}

Along with a ton of text output, you'll get a visual graph of the model's operators. My simple
classifier has this graph for example.
![Classifier compute graph](/assets/inspect_DPUCZDX8G_ISA1_B3136.png){: width="500"}
If every node says `assigned device:dpu`, you're good to go, but if not you'll have to go back
and reconfigure your model to adhere to the [supported operators](https://docs.xilinx.com/r/en-US/pg338-dpu/Introduction?tocId=4lq1FtJ078vxzAJQVMkl_g).
If the github issues are anything to go by, a common problem was support for different activation functions.
The operators table shows that only ReLU, ReLU6, LeakyReLU, Hard Sigmoid, and Hard Swish are allowed,
which notably omits the SiLU activation favored by the newer YOLO variants, or GeLU as popularized by BERT.
I'm not sure if it can be changed in place, but from my relatively shallow knowledge of deep learning,
I'd expect that the network would have to be retrained, or at least finetuned to accomodate a change
in activations.

### Calbration
Once you've got something that you know will map nicely to the DPU, you can finetune the network
using Xilinx's quantizing API. Specifically, this thing:
{% highlight python %}
quantizer = torch_quantizer(
    'calib',
    model,
    dummy_input,
    device=device,
    quant_config_file=None)

quantizer.fast_finetune(
    evaluate,
    (quantizer.quant_model, val_loader, loss_fn)
)

loss, correct = evaluate(quantizer.quant_model, val_loader, loss_fn)
{% endhighlight %}
When constructing the `torch_quantizer`, we tell it what mode to operate in ('calib'), give it our
_PyTorch model object_, the same dummy input as passed to the inspector, and the compute device to run
on. It's a bit odd, but there are several other steps to perform after that use this same function with
a different operation mode parameter. Personally I think it would have been clearer to just have separate
classes, but I digress.

Going a bit out of order, in  `fast_finetune`, `val_loader` is a dataloader for handling your calibration
dataset. The calibration set can be significantly smaller than either your training or evaluation sets,
and IIRC the recommendation was in the range of 100-1000 images, as long as they're representative of
what the model will see in practice.

I'm a bit unclear on what's strictly required for `fast_finetune` to work though; it cannot be called
with no arguments but [doc pages](https://docs.xilinx.com/r/1.3-English/ug1414-vitis-ai/Module-Partial-Quantization) and [replies by Xilinx engineers](https://github.com/Xilinx/Vitis-AI/issues/787), also say we don't need labels or a loss function, just a way to call forward().
Anyway, this `evaluate` function needs to run a forward pass of the model over your input data, and the
tuple passed in under it should match its arguments. I modeled mine after [Xilinx's resnet example](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py),
but their `evaluate` is reused in several ways. Also, `fast_finetune` is optional; running it can improve
performance, but it can be skipped entirely if you've found it makes no difference.


### Exporting

## Compiling
Once we've got our *.xmodel from the export step, the last thing to do is compile for a specific
device architecture, in this case the Kria's DPU B3136.

First go and grab the DPU fingerprint from the Kria board if you don't have it already.
You can run `xdputil query` _INSIDE_ the kria-runtime container after loading any of the Xilinx apps
like smartvision with a DPU instantiated _OUTSIDE_ the container with `xmutil loadapp kv260-nlp-smartvision`
For some reason I had to install numpy for xputil to work.
In the output you should see this line, or something like it if you're not on the Kria.
```
"fingerprint":"0x101000016010406"
```
Make a file called `arch.json` with that as the only entry; we'll need it in just a sec.

`arch.json`
```
{
    "fingerprint":"0x101000016010406"
}
```

Back in the Vitis host container, we just need to invoke the compiler with our model and DPU arch.
```
vai_c_xir -x quantize_result/MyCnn_int.xmodel -a arch.json
```
After that you should have a `deploy.xmodel`, and `md5sum.txt` in your working directory if everything
goes smoothly. If you aren't using PyTorch, you should invoke a different vai_c such as `vai_c_tensorflow2`

The last thing we need to do is make sure those weights work, so go ahead and `scp` then back to the kria,
and start an instance of kria-runtime (again) to use xdputil.
Finally `xdputil xmodel deploy.xmodel -l` should show one subgraph. We can then benchmark it.
```
root@xlnx-docker:/run/host/Documents/quantize_result# xdputil benchmark deploy.xmodel 2
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1003 23:15:44.922972   665 test_dpu_runner_mt.cpp:474] shuffle results for batch...
I1003 23:15:44.923450   665 performance_test.hpp:73] 0% ...
I1003 23:15:50.923636   665 performance_test.hpp:76] 10% ...
I1003 23:15:56.923797   665 performance_test.hpp:76] 20% ...
I1003 23:16:02.923959   665 performance_test.hpp:76] 30% ...
I1003 23:16:08.924115   665 performance_test.hpp:76] 40% ...
I1003 23:16:14.924278   665 performance_test.hpp:76] 50% ...
I1003 23:16:20.924437   665 performance_test.hpp:76] 60% ...
I1003 23:16:26.924599   665 performance_test.hpp:76] 70% ...
I1003 23:16:32.924762   665 performance_test.hpp:76] 80% ...
I1003 23:16:38.924928   665 performance_test.hpp:76] 90% ...
I1003 23:16:44.925087   665 performance_test.hpp:76] 100% ...
I1003 23:16:44.925154   665 performance_test.hpp:79] stop and waiting for all threads terminated....
I1003 23:16:44.925801   665 performance_test.hpp:85] thread-0 processes 199424 frames
I1003 23:16:44.925851   665 performance_test.hpp:85] thread-1 processes 199455 frames
I1003 23:16:44.925873   665 performance_test.hpp:93] it takes 697 us for shutdown
I1003 23:16:44.925894   665 performance_test.hpp:94] FPS= 6647.71 number_of_frames= 398879 time= 60.0024 seconds.
I1003 23:16:44.925951   665 performance_test.hpp:96] BYEBYE
Test PASS.
```

All that work and we're _still_ not done. Next time we'll add a cpp utility to run inference on an image.

## Model inspector
```
DPU Arch":"DPUCZDX8G_ISA1_B3136"
"fingerprint":"0x101000016010406"
```
