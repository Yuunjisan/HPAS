#include <iostream>
#include <catch2/catch_all.hpp>
#include <random>
#include <chrono>xxxxxxxxx

#include <ft/ft.h>

//#define RUN_BENCHMARKS // Uncomment if you would like to run the example benchmark
//#define FFT_2D_TEST_OUT

std::vector<double> generate_random_vector(size_t size) {
    std::vector<double> data(size);

    // Random data generation based on https://stackoverflow.com/questions/21102105/random-double-c11
    const double lower_bound = 0;
    const double upper_bound = 1;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);

    std::random_device rand_dev;              // Use random_device to get a random seed.

    std::mt19937 rand_engine(rand_dev()); // mt19937 is a good pseudo-random number generator

    for (int i = 0; i < size; i++) {
        data[i] = unif(rand_engine);
    }

    return data;
}


TEST_CASE("DFT Idempotence Large Random Data", "[dft-1d]") {
    auto size = GENERATE(1, 2, 4, 8, 255, 256, 500, 1024);

    auto data = generate_random_vector(size);

    NDArray<double> fromData(data);

    auto dftTestOut = dft_1d(fromData);
    auto idftTestOut = idft_1d(dftTestOut);

    REQUIRE(data.size() == dftTestOut.shape[0]);
    REQUIRE(dftTestOut.shape[0] == idftTestOut.shape[0]);

    REQUIRE(fromData.approxEquals(idftTestOut));

}

#ifdef RUN_BENCHMARKS
TEST_CASE("DFT/FFT Performance", "[dft-1d]") {
    size_t size = 1024;

    auto data = generate_random_vector(size);

    NDArray<double> fromData{ data };

    BENCHMARK("DFT of 1024 points") {
        return dft_1d(fromData);
    };
}
#endif

TEST_CASE("Recursive FFT simple known cases", "[fft]")
{
    SECTION("FFT of [1, 0, 0, 0] is [1, 1, 1, 1]")
    {
        NDArray<double> input = NDArray<double>::empty({ 4 });

        input(0) = 1.0;
        input(1) = 0.0;
        input(2) = 0.0;
        input(3) = 0.0;

        auto actual = cooley_tukey_fft_1d(input);

        for (int i = 0; i < 4; i++)
        {
            REQUIRE(std::abs(actual(i) - std::complex<double>(1.0, 0.0)) < 1e-9);
        }
    }

    SECTION("FFT of [1, 1, 1, 1] is [4, 0, 0, 0]")
    {
        NDArray<double> input = NDArray<double>::empty({ 4 });

        input(0) = 1.0;
        input(1) = 1.0;
        input(2) = 1.0;
        input(3) = 1.0;

        auto actual = cooley_tukey_fft_1d(input);

        REQUIRE(std::abs(actual(0) - std::complex<double>(4.0, 0.0)) < 1e-9);
        REQUIRE(std::abs(actual(1) - std::complex<double>(0.0, 0.0)) < 1e-9);
        REQUIRE(std::abs(actual(2) - std::complex<double>(0.0, 0.0)) < 1e-9);
        REQUIRE(std::abs(actual(3) - std::complex<double>(0.0, 0.0)) < 1e-9);
    }
}

TEST_CASE("Recursive FFT matches DFT for real-valued input", "[fft]")
{
    auto size = GENERATE(2, 4, 8, 16, 32, 64);

    NDArray<double> realInput = NDArray<double>::empty({ size });

    for (int i = 0; i < size; i++)
    {
        double value = static_cast<double>((i * 7 + 3) % 11);
        realInput(i) = value;
    }

    auto expected = dft_1d(realInput);
    auto actual = cooley_tukey_fft_1d(realInput);

    for (int i = 0; i < size; i++)
    {
        REQUIRE(std::abs(actual(i) - expected(i)) < 1e-9);
    }
}

TEST_CASE("Recursive FFT followed by IFFT reconstructs input", "[fft]")
{
    auto size = GENERATE(2, 4, 8, 16, 32, 64);

    NDArray<double> input = NDArray<double>::empty({ size });

    for (int i = 0; i < size; i++)
    {
        input(i) = static_cast<double>((i * 7 + 3) % 11);
    }

    auto frequency = cooley_tukey_fft_1d(input);
    auto reconstructedComplex = cooley_tukey_ifft_1d(frequency);

    for (int i = 0; i < size; i++)
    {
        double reconstructed = reconstructedComplex(i).real() / size;

        REQUIRE(std::abs(reconstructed - input(i)) < 1e-9);
        REQUIRE(std::abs(reconstructedComplex(i).imag()) < 1e-9);
    }
}

TEST_CASE("Iterative FFT matches DFT for real-valued input", "[fft]")
{
    auto size = GENERATE(2, 4, 8, 16, 32, 64);

    NDArray<double> input = NDArray<double>::empty({ size });

    for (int i = 0; i < size; i++)
    {
        double value = static_cast<double>((i * 7 + 3) % 11);
        input(i) = value;
    }

    auto expected = dft_1d(input);
    auto actual = iterative_fft_1d(input);

    for (int i = 0; i < size; i++)
    {
        REQUIRE(std::abs(actual(i) - expected(i)) < 1e-9);
    }
}

TEST_CASE("Iterative FFT followed by IFFT reconstructs input", "[fft]")
{
    auto size = GENERATE(2, 4, 8, 16, 32, 64);

    NDArray<double> input = NDArray<double>::empty({ size });

    for (int i = 0; i < size; i++)
    {
        input(i) = static_cast<double>((i * 7 + 3) % 11);
    }

    auto frequency = iterative_fft_1d(input);
    auto reconstructedComplex = iterative_ifft_1d(frequency);

    for (int i = 0; i < size; i++)
    {
        double reconstructed = reconstructedComplex(i).real() / size;

        REQUIRE(std::abs(reconstructed - input(i)) < 1e-9);
        REQUIRE(std::abs(reconstructedComplex(i).imag()) < 1e-9);
    }
}

TEST_CASE("2D iterative FFT and IFFT round trip", "[fft-2d]")
{
    size_t size = 64;

    auto data = generate_random_vector(size);

    NDArray<double> input(data);
    input.reshape(8, 8);

    auto frequency = fft_2d(input, true);
    auto reconstructed = ifft_2d(frequency, true);

    REQUIRE(reconstructed.approxEquals(input));
}

TEST_CASE("2D recursive FFT and IFFT round trip", "[fft-2d]")
{
    size_t size = 64;

    auto data = generate_random_vector(size);

    NDArray<double> input(data);
    input.reshape(8, 8);

    auto frequency = fft_2d(input, false);
    auto reconstructed = ifft_2d(frequency, false);

    REQUIRE(reconstructed.approxEquals(input));
}

TEST_CASE("2D DFT Idempotence, square", "[dft-2d]") {
    size_t size = 64;

    auto data = generate_random_vector(size);

    NDArray<double> in(data);
    in.reshape(8, 8);

#ifdef FFT_2D_TEST_OUT
    std::cout << "Input data:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << in(i, j) << " ";
        }
    }
#endif

    auto dftTestOut = dft_2d(in);

#ifdef FFT_2D_TEST_OUT
    std::cout << "Fourier domain of data:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << dftTestOut(i, j) << " ";
        }
    }
    std::cout << std::endl;
#endif
    auto idftTestOut = idft_2d(dftTestOut);

#ifdef FFT_2D_TEST_OUT
    std::cout << "IFFT output:\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << idftTestOut(i, j) << " ";
        }
    }
    std::cout << std::endl;
#endif

    REQUIRE(idftTestOut.approxEquals(in));

}

TEST_CASE("FFT empirical runtime experiment", "[fft-benchmark]")
{
    std::vector<int> sizes = {
        16, 32, 64, 128, 256, 512, 1024,
        2048, 4096, 8192, 16384, 32768, 65536
    };

    constexpr int repetitions = 20;

    std::cout << "\nFFT Runtime Experiment\n";
    std::cout << "n,recursive_ms,iterative_ms\n";

    for (int size : sizes)
    {
        auto data = generate_random_vector(size);
        NDArray<double> input(data);

        // Warm-up
        auto warmupRecursive = cooley_tukey_fft_1d(input);
        auto warmupIterative = iterative_fft_1d(input);

        auto recursiveStart = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < repetitions; r++)
        {
            auto result = cooley_tukey_fft_1d(input);
        }

        auto recursiveEnd = std::chrono::high_resolution_clock::now();

        auto iterativeStart = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < repetitions; r++)
        {
            auto result = iterative_fft_1d(input);
        }

        auto iterativeEnd = std::chrono::high_resolution_clock::now();

        double recursiveMs =
            std::chrono::duration<double, std::milli>(recursiveEnd - recursiveStart).count()
            / repetitions;

        double iterativeMs =
            std::chrono::duration<double, std::milli>(iterativeEnd - iterativeStart).count()
            / repetitions;

        std::cout << size << ","
            << recursiveMs << ","
            << iterativeMs << "\n";
    }
}
