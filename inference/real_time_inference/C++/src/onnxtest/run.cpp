#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <time.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    const wchar_t* model_path = L"./model.onnx";
#else
    const char* model_path = "dpcrn.onnx";
#endif
    int step=0;
    std::cout << "input the step:"<<std::endl;
    std::cin >> step;

    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names = {"input","h_in"};
    std::vector<const char*> output_node_names = {"output","h_out"};
    // get input tensor
    std::vector<int64_t> input_node_dims = {1, 3, 1, 257};
    size_t input_tensor_size = 3 * 257; 
    std::vector<float> input_tensor_values(input_tensor_size);
    
    for (unsigned int i = 0; i < input_tensor_size; i++){
        input_tensor_values[i] = (float)i / input_tensor_size;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    // get input hidden states
    std::vector<int64_t> input_hidden_node_dims = {2, 32, 128};
    size_t input_hidden_tensor_size = 2 * 32 * 128; 
    std::vector<float> input_hidden_tensor_values(input_hidden_tensor_size);

    for (unsigned int i = 0; i < input_hidden_tensor_size; i++){
        input_hidden_tensor_values[i] = (float)i / (input_hidden_tensor_size + 1);
    }

    auto hidden_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_hidden_tensor = Ort::Value::CreateTensor<float>(hidden_memory_info, input_hidden_tensor_values.data(), input_hidden_tensor_size, input_hidden_node_dims.data(), 3);
    assert(input_hidden_tensor.IsTensor());

    // inference
    clock_t start, end;
    //std::vector<float> time;

    std::vector<Ort::Value> ort_inputs;
    std::vector<Ort::Value> output_tensors;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(input_hidden_tensor));

    float *output, *output_hidden;
    /*
    session.Run(run_options, input_names, input_values, input_count, output_names, output_count)
    OrtRun(session_, nullptr, input_names, &input_tensor, input_count, output_names, output_count, &output_tensor);
    */
    start = clock();
    for(int i =0 ; i < step; i++){

      fill(input_tensor_values.begin(), input_tensor_values.end(), (float)i);
      output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
      // get pointer to output tensor float values
      output = output_tensors[0].GetTensorMutableData<float>(); 
      output_hidden = output_tensors[1].GetTensorMutableData<float>();
      //usleep(16000);
    }

    end = clock(); 
    for(int i =0;i<20;i++){
    std::cout << i << " " << output[i] << std::endl;
    }
    std::cout<< (double)(end - start) / CLOCKS_PER_SEC / step * 1000 << " ms/frame" << std::endl;

    std::cout<< input_node_names.data()[0] << std::endl;
    printf("Done!\n");  
}
