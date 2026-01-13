#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <unistd.h> // getpid
 
static const std::vector<double> MEAN = {0.4914, 0.4822, 0.4465};
static const std::vector<double> STD  = {0.2470, 0.2435, 0.2616};
 
torch::Tensor normalize(torch::Tensor x) {
  x = x.to(torch::kFloat32).div_(255.0);
  for (int c = 0; c < 3; ++c) {
    x.index({"...", c, torch::indexing::Slice(), torch::indexing::Slice()})
      .sub_(MEAN[c]).div_(STD[c]);
  }
  return x;
}
 
int main() {
  std::cout << "PID=" << getpid() << std::endl;
 
  torch::jit::script::Module module = torch::jit::load("../artifacts/scripted_model.pt");
  module.eval();
 
  const int N = 8;
  auto input = torch::randint(0, 256, {N, 3, 32, 32}, torch::kByte);
  input = normalize(input).to(torch::kFloat32);
 
  // --- set a GDB breakpoint on this next line later ---
  auto out = module.forward({input}).toTensor();
 
  auto pred = std::get<1>(out.max(1));
  std::cout << "out shape: " << out.sizes() << "\n";
  std::cout << "pred[0]: " << pred[0].item<int>() << "\n";
  std::cout << std::hex << "input data_ptr: 0x" << (uint64_t)input.data_ptr() << std::dec << "\n";
  return 0;
}