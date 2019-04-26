use super::fft::*;
use num_complex::*;
use super::allocate_aligned_vec_128;
use std::vec::Vec;
use libm::F32Ext;
use std::time::SystemTime;
use rand::Rng;

//  RUSTFLAGS="-C target-feature=+avx2" cargo run --release
pub fn testing_fft(num_subcarriers : u32) {
    let N: u32 = num_subcarriers;
    let mut rng = rand::thread_rng();

    // allocate vectors for fast fft
    let mut input: Vec<Complex32> = Vec::new();
    let mut output: Vec<Complex32> = Vec::new();
    allocate_aligned_vec_128(N as usize, &mut input);
    allocate_aligned_vec_128(N as usize, &mut output);

    // allocate vectors for basic fft, as a comparison
    let mut in_out: Vec<Complex32> = Vec::new();
    allocate_aligned_vec_128(N as usize, &mut in_out);

    // initialize the input with random numbers
    for sc in 0..N {
        let r = rng.gen();
        let i = rng.gen();

        input[sc as usize].re = r ; input[i as usize].im = i;
        in_out[sc as usize].re = r ; in_out[i as usize].im = i; 
    }

    let plan = mufft_create_plan_1d_c2c(N, MUFFT_FORWARD, 0);

    match plan {
        Some(x) => {
            // execute both of the ffts
            mufft_execute_plan_1d(&x, &mut output, &mut input);
            fft_basic(&mut in_out);


            let mut pass = true;
            let epsilon : f32 = 0.001;// random

            // compare outputs
            for i in 0..N
            {
                let delta_c = output[i as usize] - in_out[i as usize];
                if (delta_c.re > epsilon) | (delta_c.re < -epsilon) | (delta_c.im > epsilon) | (delta_c.im < -epsilon) {
                    pass = false;
                }
            } 

            // print if the fast fft is accurate
            if pass == false {
                println!("mufft fails");
            }
            else {
                println!("mufft passed");
            }
            
        },

        None => println!("There's no plan :("),
    }
}


pub fn fft_basic(input: &mut Vec<Complex<f32>>) {
    //let mut output = Symbol{data: [Complex{real: 0.0, imag: 0.0};NUM_SUBCARRIERS] }; 
    let pi: f32 = 3.1415926;

	let N = input.len();
	if N <= 1 {
		return;
	}

	//divide
	let mut even : Vec<Complex<f32>> = Vec::with_capacity(N/2);
	let mut odd : Vec<Complex<f32>> = Vec::with_capacity(N/2);
	
	divide_array(0, N/2, 2, input, &mut even);
	divide_array(1, N/2, 2, input, &mut odd);

	fft_basic(&mut even);
	fft_basic(&mut odd);

	//combine
	let mut k = 0;
	while k < N/2 {
		
		let t = Complex{ re: F32Ext::cos(-2.0*pi*(k as u32 as f32)/  (N as u32 as f32)), im: F32Ext::sin(-2.0*pi*(k as u32 as f32) / (N as u32 as f32))} * odd[k];
		//let t = Complex {real: 0.0, imag: 0.0};
		input[k] = even[k] + t;
		input[k + N/2] = even[k] -t;

		k += 1;
	}
}

fn divide_array(start: usize, size: usize, stride: usize, input: &Vec<Complex<f32>>, output: &mut Vec<Complex<f32>>) {
	
	for i in 0..size {
		output.push(input[start + (i*stride)]);	
	}

}

pub fn bm_fft(num_subcarriers : u32) {
    let iter = 10000;
    let mut rng = rand::thread_rng();

    let N: u32 = num_subcarriers;
    let mut input: Vec<Complex32> = Vec::new();
    let mut output: Vec<Complex32> = Vec::new();

    allocate_aligned_vec_128(N as usize, &mut input);
    allocate_aligned_vec_128(N as usize, &mut output);

    for i in 0..N {
        input[i as usize].re = rng.gen() ; input[i as usize].im = rng.gen();
    }

    let plan = mufft_create_plan_1d_c2c(N, MUFFT_FORWARD, 0).unwrap();

    let start = SystemTime::now();
    for _ in 0..iter {
        mufft_execute_plan_1d(&plan, &mut output, &mut input);
    }
    let duration = start.elapsed();

    println!("time = {:?}", duration.unwrap());            
}