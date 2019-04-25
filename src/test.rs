use super::fft::*;
use num_complex::*;
use super::allocate_aligned_vec_32;
use alloc::vec::Vec;
use alloc::boxed::Box;

pub fn testing_fft(num_subcarriers : u32) {
    let N: u32 = num_subcarriers;
    let mut input: Vec<Complex32> = Vec::new();
    let mut in_out: Vec<Complex32> = Vec::new();
    let mut output: Vec<Complex32> = Vec::new();

    allocate_aligned_vec_32(N as usize, &mut input);
    allocate_aligned_vec_32(N as usize, &mut in_out);
    allocate_aligned_vec_32(N as usize, &mut output);

    for i in 0..N {
        input[i as usize].re = 1.0 ; input[i as usize].im = 0.0;
        in_out[i as usize].re = 1.0 ; in_out[i as usize].im = 0.0; 
    }

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


