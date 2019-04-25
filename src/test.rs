use super::fft::*;
use num_complex::*;
use super::trig::*;
use super::allocate_aligned_vec_32;
use alloc::vec::Vec;
use alloc::boxed::Box;
use acpi::get_hpet;

// #[cfg(target_feature = "sse2")]
pub fn testing_fft(num_subcarriers : u32) {
    let N: u32 = num_subcarriers;
    let mut input: Vec<Complex32> = Vec::new();
    let mut in_out: Vec<Complex32> = Vec::new();
    let mut output: Vec<Complex32> = Vec::new();

    allocate_aligned_vec_32(N as usize, &mut input);
    allocate_aligned_vec_32(N as usize, &mut in_out);
    allocate_aligned_vec_32(N as usize, &mut output);

    // let mut input : Box<Vec<Complex32>> = Box::new(a);
    // let mut in_out : Box<Vec<Complex32>> = Box::new(b);
    // let mut output : Box<Vec<Complex32>> = Box::new(c);

    for i in 0..N {
        input[i as usize].re = 1.0 ; input[i as usize].im = 0.0;
        in_out[i as usize].re = 1.0 ; in_out[i as usize].im = 0.0; 
    }

    // input[2].re = 0.5;
    // input[2].im = 0.07;

    // in_out[2].re = 0.5;
    // in_out[2].im = 0.07;

    for _ in 0..N{
        output.push(Complex32{re: 0.0, im: 0.0});
    }

    let plan = mufft_create_plan_1d_c2c(N, MUFFT_FORWARD, 0);

    match plan {
        Some(mut x) => {
            debug!("There's a plan!!!");
            mufft_execute_plan_1d(&x, &mut output, &mut input);
            debug!("completed mufft");
            fft_basic(&mut in_out);
            let mut pass = true;
            let epsilon : f32 = 0.001;// random
            
            // debug!("mufft");
            // for i in 0..N {
            //     debug!("{}", output[i as usize]);
            // }

            // debug!("fft");
            // for i in 0..N {
            //     debug!("{}", in_out[i as usize]);
            // }

            // debug!("delta");
            for i in 0..N
            {
                let delta_c = output[i as usize] - in_out[i as usize];
                // debug!("{}", delta_c);
                if (delta_c.re > epsilon) | (delta_c.re < -epsilon) | (delta_c.im > epsilon) | (delta_c.im < -epsilon) {
                    pass = false;
                }
            } 

            if pass == false {
                debug!("mufft fails");
            }
            else {
                debug!("mufft passed");
            }
            
        },

        None => debug!("There's no plan :("),
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
		
		let t = Complex{ re: cos(-2.0*pi*(k as u32 as f32)/  (N as u32 as f32)), im: sin(-2.0*pi*(k as u32 as f32) / (N as u32 as f32))} * odd[k];
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

pub fn captain_fft(_:()){
    debug!("IDK why but application only works when I run a function in Captain crate");
}
/* #[cfg(target_feature = "sse2")]
pub fn test_mufft(_ : Option<u64>) {

    let twiddles = build_twiddles(64, -1);

    let mut input : Vec<Complex32> = Vec::with_capacity(64);
    let mut output : Vec<Complex32> = Vec::with_capacity(64);

    for _ in 0..64 {
        input.push(Complex{re: 1.0, im: 1.0});
        output.push(Complex{re: 0.0, im: 0.0});
    }

    mufft_radix8_generic(&mut output, &input, &twiddles, 8, 64);

    /* for i in 0..64 {
        debug!("{}",output[i]);
    } */

}



pub fn test_fft(_:Option<u64>) {
    
    
    let twiddles = build_twiddles(64, -1);

    let mut input : Vec<Complex32> = Vec::with_capacity(64);
    let mut output : Vec<Complex32> = Vec::with_capacity(64);

    for _ in 0..64 {
        input.push(Complex{re: 1.0, im: 1.0});
        output.push(Complex{re: 0.0, im: 0.0});
    }
	
	let start1 = get_hpet().as_ref().unwrap().get_counter();
    //mufft_radix8_generic(&mut output, &input, &twiddles, 8, 64);
	let end1 = get_hpet().as_ref().unwrap().get_counter();

    let mut input1:  Vec<Complex<f32>> = Vec::with_capacity(64);
    for _ in 0..64 {
        input1.push(Complex{re: 1.0, im: 1.0});
    }
	let start2 = get_hpet().as_ref().unwrap().get_counter();
	fft_basic(&mut input1);
	let end2 = get_hpet().as_ref().unwrap().get_counter();

	debug!("RustFFT: {}", end1-start1);
	debug!("BasicFFT: {}", end2-start2);

} */