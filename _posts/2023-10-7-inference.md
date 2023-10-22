---
layout: post
title:  "Custom model inference on VART"
date:   2023-10-01 12:27:00 -1000
categories: xilinx edge_ai inference vitis
---
## Model inference
In the last post, we managed to quantize a PyTorch model with Vitis-AI, and export its int8
weights and graph for use with Vitis AI Runtime (VART). Now, we still need some code to drive
VART, so let's look at that now.

### A note about Vitis AI levels
VART has several levels of abstraction they call API_x that make different assumptions
about your model. The levels 1 and 2 appear to work only with models matching one of
the architectures in their model zoo, while level 0 is more work to use but can work with just abot
anything, and level 3 is required for models with custom ops or multiple subgraphs.
Check out the [full docs here](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/Programming-Examples).

The API_0 sample is in `Vitis-AI/examples/vai_runtime/resnet50/src/main.cc`, so we'll be
basing our code off of that and some [VART documentation](https://xilinx.github.io/Vitis-AI/3.5/html/doxygen/api/class/classvart_1_1_runner.html)


### Code sample
I'll be referring to [this file](https://github.com/kyflores/kv260-vitis-flow/blob/main/cpp/main.cc).
Apologies in advance for some janky c++...Maybe if you're reading this in the future I'll
have cleaned this up a bit, but some of the VART API stuff threw me for a loop.
my goal here was to get everything that matters into one file, which almost succeeded, except
for a dependency on [stb_image](https://github.com/nothings/stb) for reading images. If you don't
know about `stb`, it's definitely worth a look for single-file, header only code for common tasks.

This all starts off by reading an image (which I separately pulled from the FashionMNIST dataset),
and the xmodel file we produced last time. There's several subgraphs in the xmodel file, but only
one (for this network) will have the device attribute set to DPU.

{% highlight c++ %}
static constexpr const char* MODELNAME = "quantize_result/deploy.xmodel";
auto image = tensor_from_stbimage(argv[1]);

auto graph = xir::Graph::deserialize(MODELNAME);

auto graph_root = graph.get()->get_root_subgraph();
auto children = graph_root->children_topological_sort();

// Check subgraphs on CLI with `xdputil xmodel quantize_result/deploy.xmodel -l`
const xir::Subgraph* subgraph;
for (auto c : children) {
    auto device = c->get_attr<std::string>("device");
    std::cout << "Device: " << device << std::endl;
    std::cout << c->get_name() << std::endl;
    if (device == "DPU") {
        subgraph = c;
    }
}
auto runner = vart::Runner::create_runner(subgraph, "run");
{% endhighlight %}

In this section we get a handle of sorts to the input and output tensors from the graph
This API confused me a bit; I'm used to `tensor` storing both the metadata like shape as well
as the data like in Torch, but this VART tensor class seems to be only a description of what
sort of data is required at the inputs and outputs. There's a separate `TensorBuffer` class
in VART that holds the data, and we'll see that a few lines down.

`get_scale` is a helper function up above that gets the value of the `fix_point` attribute in
the input and output tensors and converts it to a scale factor. This fix_point value tells us
how to convert a floating point value to an int8 in the range this model needs at the inputs and
outputs. Keep in mind for a moment too that at the input, we're converting float->int8, and at
the output we're doing int8->float, so at the output we need to divide by the scale factor instead.

{% highlight c++ %}
auto input_tensors = runner->get_input_tensors();
float in_scale = get_scale(input_tensors[0]);
auto output_tensors = runner->get_output_tensors();
float out_scale = get_scale(output_tensors[0]);

std::cout << "Input scale " << in_scale << std::endl;
std::cout << "Output scale " << out_scale << std::endl;
{% endhighlight %}

The next step is to create TensorBuffers that will hold the actual inputs and outputs.
We're using this `vart::alloc_cpu_flat_tensor_buffer(input)` helper function to do this
for us. `TensorBuffer` is actually a base class in VART, and I assume the classes derived from
it hold buffers in different locations, whether that's main RAM, or an HBM stack private
to the FPGA. Since the Kria's FPGA doesn't have its own dedicated memory, we're sharing the
CPU's 4GB memory here, thus CpuFlatTensorBuffer. This TensorBuffer has a bit of an awkward
API though, not sure why `CpuFlatTensorBuffer::data()` needs to return its data pointer as a
u64 that we need to reinterpret cast back to actually use. Xilinx holds it as a `void*` inside
the class and reinterpret_casts it before returning it.

One other note here is that the training and quantiziation code scaled all the image input
values to be between 0 and 1. The inference code needs to do the same so that the calibrated
ranges are valid.

{% highlight c++ %}
for (auto input : input_tensors) {
    auto t = vart::alloc_cpu_flat_tensor_buffer(input);

    // Copy data by ptr into the buffer
    auto tensor_data = t->data();
    // This CpuFlatTensorBuffer class returns a uint64_t by reinterpret casting its 64bit data
    // pointer into an integer, wtf? Need to undo that here to get something we can use as a ptr
    int8_t* raw_ptr_int = reinterpret_cast<int8_t*>(std::get<0>(tensor_data));
    size_t raw_bytes = std::get<1>(tensor_data);
    std::cout << "Input tensor is " << raw_bytes << " bytes long" << std::endl;
    for (int x = 0; x < raw_bytes; ++x) {
        // In our training code, the uint8 images are squished to the range (0, 1)
        // before being input to the network, so replicate that here.
        float tmp = static_cast<float>(image[x]) / 256;

        raw_ptr_int[x] = static_cast<int8_t>(tmp * in_scale);
    }

    input_tensor_buffers.emplace_back(std::move(t));
}
{% endhighlight %}

Before launching the DPU, we need to call `sync_for_write` on our input, then gather
the TensorBuffer pointers together for `execute_async`. This looks kind of hacky to me,
so I might be using the API wrong here...but up above `alloc_cpu_flat_tensor_buffer` returns
its buffers behind a unique_ptr, but execute_async requires a raw pointer.

Then we wait for the computation to finish and finally sync the output buffer.
{% highlight c++ %}
for (auto& input : input_tensor_buffers) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
            input->get_tensor()->get_shape()[0]);
}
std::cout << "Executing runner..." << std::endl;
std::vector<vart::TensorBuffer*> input_ptrs;
for (auto& ptr : input_tensor_buffers) { input_ptrs.push_back(ptr.get()); }
std::vector<vart::TensorBuffer*> output_ptrs;
for (auto& ptr : output_tensor_buffers) { output_ptrs.push_back(ptr.get()); }

auto v = runner->execute_async(input_ptrs, output_ptrs);
auto status = runner->wait((int)v.first, 1000000000);

for (auto& output : output_tensor_buffers) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
    output->get_tensor()->get_shape()[0]);
}
{% endhighlight %}

We're basically done at this point. To wrap up we just run softmax over the outputs.
Unfortunately there's no built softmax operator on the DPU, and I didn't see one in
VART either, so here's a naive softmax implementation to use instead. Note that this
won't work very well if your inputs get too large...
{% highlight c++ %}
void print_softmax(int8_t* data, size_t len, float scale) {
    double denom = 0;
    std::vector<double> vec;
    for (int i = 0; i < len; ++i) {
        float tmp = static_cast<double>(data[i] / scale);
        tmp = std::exp(tmp);
        denom += tmp;
        vec.push_back(tmp);
    }

    std::cout << std::endl;
    for (int x = 0; x < len; ++x) {
        std::cout << (vec[x] / denom) << " ";
    }
    std::cout << std::endl;
}

auto& out_buf = output_tensor_buffers.back();
auto out_data = out_buf->data();
int8_t* raw_ptr_int = reinterpret_cast<int8_t*>(std::get<0>(out_data));
size_t raw_bytes = std::get<1>(out_data);
std::cout << "Output tensor is " << raw_bytes << " bytes long" << std::endl;
print_softmax(raw_ptr_int, raw_bytes, out_scale);
{% endhighlight %}

At long last...our results! This image was from label index 1 (from 0 to 9),
and we can see at the bottom that we predicted 99.63% confidence for label 1.

![sample](/assets/fmnist_1.jpg){: width="250" }
```
x, y, n: 28 28 1
Device: USER
subgraph_MyCnn__input_0
Device: DPU
subgraph_MyCnn__MyCnn_Conv2d_conv1_1__ret_3
Device: CPU
subgraph_MyCnn__MyCnn_Linear_lin2__ret_fix_
Input scale 64
Output scale 8
Populating input tensor
1 Input(s)
Input tensor is 784 bytes long
1 Output(s)
Executing runner...
Output tensor is 10 bytes long

3.52274e-05 0.996319 2.55186e-06 0.00359329 4.52328e-05 9.20345e-09 3.71294e-06 2.20779e-08 6.45211e-07 1.63135e-07
```
