#include <iostream>
#include "sovereign_kernel.h"

int main()
{
    std::cout << "--- SovereignKernel Tensor Stress Test ---\n";

    Tensor x({4}, Device::CPU);

    x.data()[0] = 1.0f;
    x.data()[1] = 2.0f;
    x.data()[2] = 3.0f;
    x.data()[3] = 4.0f;


    std::cout << "Tensor values:\n";

    for (int i = 0; i < 4; i++)
    {
        std::cout << x.data()[i] << " ";
    }

    std::cout << "\n";


    float sum = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        sum += x.data()[i];
    }


    float mean = sum / 4.0f;


    std::cout << "Sum:  " << sum << "\n";
    std::cout << "Mean: " << mean << "\n";


    bool pass =
        sum == 10.0f &&
        mean == 2.5f;


    if(pass)
        std::cout << "[PASS] Tensor memory verified\n";
    else
        std::cout << "[FAIL] Tensor calculation error\n";


    return pass ? 0 : 1;
}