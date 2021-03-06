use num_complex::*;
// use std::vec::Vec;
// use std::boxed::Box;

use super::kernel::*;
use super::allocate_aligned_vec_128;
use libm::F32Ext;
use std::arch::x86_64::{_mm256_load_ps, _mm256_store_ps};

/// The forward FFT transform.
pub const MUFFT_FORWARD: i32 = -1;
/// The inverse FFT transform.
pub const MUFFT_INVERSE: i32 = 1;
/// muFFT will use any SIMD instruction set it can if supported by the CPU.
pub const MUFFT_FLAG_CPU_ANY: u32 = 0;
/// muFFT will not use any SIMD instruction set.
pub const MUFFT_FLAG_CPU_NO_SIMD: u32 = ((1 << 16) - 1);
/// muFFT will not use the AVX instruction set.
pub const MUFFT_FLAG_CPU_NO_AVX: u32 = (1 << 0);
/// muFFT will not use the SSE3 instruction set.
pub const MUFFT_FLAG_CPU_NO_SSE3: u32 = (1 << 1);
/// muFFT will not use the SSE instruction set.
pub const MUFFT_FLAG_CPU_NO_SSE: u32 = (1 << 2);
/// The real-to-complex 1D transform will also output the redundant conjugate values X(N - k) = X(k)*.
pub const MUFFT_FLAG_FULL_R2C: u32 = (1 << 16);
/// The second/upper half of the input array is assumed to be 0 and will not be read and memory for the second half of the input array does not have to be allocated.
/// This is mostly useful when you want to do zero-padded FFTs which are very common for convolution-type operations, see \ref MUFFT_CONV. This flag is only recognized for 1D transforms.
pub const MUFFT_FLAG_ZERO_PAD_UPPER_HALF: u32 = (1 << 17);
/// Internal flag used for choosing FFT routines
pub const MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF: u32 = (1 << 28);
/// Internal flag used for choosing FFT routines
pub const MUFFT_FLAG_DIRECTION_INVERSE: u32 = (1 << 24);
/// Internal flag used for choosing FFT routines
pub const MUFFT_FLAG_DIRECTION_FORWARD: u32 = (1 << 25);
/// Internal flag used for choosing FFT routines
pub const MUFFT_FLAG_DIRECTION_ANY: u32 = 0;

/// Size of FFT 1D table
pub const SIZE_FFT_1D_TABLE: usize = 13;
/// min_x when using sse.. see line 253 in fft.c of original file
pub const MIN_X: u32 = 2;
pub const MUFFT_HAVE_SSE: u32 = 1;
pub const M_PI: f32 = 3.14159265358979323846;

pub const U_NEG1: u32 = 4294967295; //replace -1u in c

/// Represents a single step of a complete 1D/horizontal FFT.
pub struct mufft_step_1d
{
    pub func: u8, //< Function pointer to a 1D partial FFT.
    pub radix: u32, //< Radix of the FFT step. 2, 4 or 8.
    pub p: u32, //< The current p factor of the FFT. Determines butterfly stride. It is equal to prev_step.p * prev_step.radix. Initial value is 1.
    pub twiddle_offset: u32, //< Offset into twiddle factor table.
}

/// Represents a single step of the complete 1D/horizontal FFT with requirements on use.
pub struct fft_step_1d
{
    pub func: u8, // < Function pointer to a 1D partial FFT.
    pub radix: u32, //< Radix of the FFT step. 2, 4 or 8.
    pub minimum_elements: u32, //< Minimum transform size for which this function can be used.
    pub fixed_p: u32, //< Non-zero if this can only be used with a fixed value for mufft_step_base::p.
    pub minimum_p: u32, //< Minimum p-factor for which this can be used. Set to -1u if it can only be used with fft_step_1d::fixed_p.
    pub flags: u32, //< Flags which determine under which conditions this function can be used.
}

pub static fft_1d_table: [fft_step_1d; SIZE_FFT_1D_TABLE] = [
    fft_step_1d {func: 0, radix: 8, minimum_elements: 8 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF}, //64,1024
    
    fft_step_1d {func: 1, radix: 4, minimum_elements: 4 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF},
    
    fft_step_1d {func: 2, radix: 2, minimum_elements: 2 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_ANY | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF},

    fft_step_1d {func: 3, radix: 8, minimum_elements: 8 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_ZERO_PAD_UPPER_HALF},

    fft_step_1d {func: 4, radix: 4, minimum_elements: 4 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_ZERO_PAD_UPPER_HALF},

    fft_step_1d {func: 5, radix: 2, minimum_elements: 2 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_ANY | MUFFT_FLAG_ZERO_PAD_UPPER_HALF},

    fft_step_1d {func: 6, radix: 2, minimum_elements: 2 * MIN_X, fixed_p: 2, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_FORWARD},

    fft_step_1d {func: 7, radix: 8, minimum_elements: 8 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_INVERSE},
    
    fft_step_1d {func: 8, radix: 4, minimum_elements: 4 * MIN_X, fixed_p: 1, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_INVERSE},
    
    fft_step_1d {func: 9, radix: 2, minimum_elements: 2 * MIN_X, fixed_p: 2, minimum_p: U_NEG1, 
        flags: MUFFT_FLAG_DIRECTION_INVERSE},

    fft_step_1d {func: 10, radix: 8, minimum_elements: 8 * MIN_X, fixed_p: 0, minimum_p: 8, 
        flags: MUFFT_FLAG_DIRECTION_ANY}, //64, 1024

    fft_step_1d {func: 11, radix:4, minimum_elements: 4 * MIN_X, fixed_p: 0, minimum_p: 4, 
        flags: MUFFT_FLAG_DIRECTION_ANY}, 
    
    fft_step_1d {func: 12, radix:2, minimum_elements: 2 * MIN_X, fixed_p: 0, minimum_p: 4, 
        flags: MUFFT_FLAG_DIRECTION_ANY},//1024

];

/// Represents a complete plan for a 1D FFT.
pub struct mufft_plan_1d
{
    pub steps: Vec<mufft_step_1d>, //< A list of steps to take to complete a full N-tap FFT.
    pub num_steps: u32, //< Number of steps contained in mufft_plan_1d::steps.
    pub N: u32, //< Size of the 1D transform.

    pub tmp_buffer: Box<Vec<Complex32>>, //< A temporary buffer used during intermediate steps of the FFT.
    pub twiddles: Vec<Complex32>, //< Buffer holding twiddle factors used in the FFT.
}


/// Computes the twiddle factor exp(pi * I * direction * k / p)
pub fn twiddle(direction: i32, k: u32, p: u32) -> Complex32
{
    let phase = (M_PI * direction as f32 * k as f32) / p as f32;
    return Complex {re: F32Ext::cos(phase), im: F32Ext::sin(phase)};
}

pub fn build_twiddles(n: u32, direction: i32, mut twiddles: &mut Vec<Complex32>)
{
    let _ = allocate_aligned_vec_128(n as usize, &mut twiddles);
    let mut p: u32 = 1; 
    let mut j: u32 = 0;

    while p < n 
    {
        for k in 0..p
        {
            twiddles[(k + j) as usize] = twiddle(direction, k, p);
        }
        
        if p == 2 { j += 3; }
        else { j += p;}

        p = p << 1;
    }
}



/// \brief Adds a new FFT step to either \ref mufft_step_1d or \ref mufft_step_2d.
pub fn add_step(steps: &mut Vec<mufft_step_1d>, num_steps: &mut u32, step: & fft_step_1d, p: u32) -> bool
{
    let mut twiddle_offset: u32 = 0;
    if *num_steps != 0
    {
        let prev = &steps[(*num_steps as usize) - 1];

        if prev.p == 2 {
            twiddle_offset = prev.twiddle_offset + 3; 
        }
        else { 
            twiddle_offset = prev.twiddle_offset + prev.p * (prev.radix - 1);
        }

        // We skipped radix2 kernels, we have to add the padding twiddle here.
        if p >= 4 && prev.p == 1
        {
            twiddle_offset += 1;
        }
    }

    steps.push (mufft_step_1d{
        func: step.func,
        radix: step.radix,
        p: p,
        twiddle_offset: twiddle_offset,
    });

    *num_steps += 1;

    return true;
}

/// \brief Builds a plan for a horizontal transform.
pub fn build_plan_1d(steps: &mut Vec<mufft_step_1d> , num_steps: &mut u32, N: u32, direction: i32, flags: u32) -> bool
{
    let mut radix: u32 = N;
    let mut p: u32 = 1;

    let mut step_flags: u32 = 0;

    match direction
    {
        MUFFT_FORWARD => step_flags |= MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF,

        MUFFT_INVERSE => step_flags |= MUFFT_FLAG_DIRECTION_INVERSE | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF,

        _ => println!("wrong direction value"),

    }

    // Add CPU flags. Just accept any CPU for now, but mask out flags we don't want.
    /* step_flags |= mufft_get_cpu_flags() & ~(MUFFT_FLAG_CPU_NO_SIMD & flags);
    step_flags |= (flags & MUFFT_FLAG_ZERO_PAD_UPPER_HALF) != 0 ?
        MUFFT_FLAG_ZERO_PAD_UPPER_HALF : MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF; */
    
    while radix > 1
    {
        let mut found: bool = false;

        // Find first (optimal?) routine which can do work.
        for i in 0..SIZE_FFT_1D_TABLE {
            let mut step = & fft_1d_table[i];

            if radix % step.radix == 0 &&
                    N >= step.minimum_elements &&
                    /*(step_flags & step.flags) == step.flags &&*/
                    (p >= step.minimum_p || p == step.fixed_p)
            {
                // Ugly casting, but add_step_1d and add_step_2d are ABI-wise exactly the same, and we don't have templates :(
                if add_step(steps, num_steps, &step, p)
                {
                    found = true;
                    radix /= step.radix;
                    p *= step.radix;
                    break;
                }
            }
        }

        if !found
        {
            return false;
        }
    }

    return true;
}

pub fn mufft_create_plan_1d_c2c(N: u32, direction: i32, flags: u32) -> Option<mufft_plan_1d>
{
    if (N & (N - 1)) != 0 || N == 1
    {
        return None;
    }

    let mut plan = mufft_plan_1d {
        steps: Vec::with_capacity(N as usize), // don;t know this before hand
        num_steps: 0, 
        N: 0, 
        tmp_buffer: Box::new(Vec::with_capacity(N as usize)), 
        twiddles: Vec::with_capacity(1), // don't really need this
    };


    build_twiddles(N, direction, &mut plan.twiddles);

    if !build_plan_1d(&mut plan.steps, &mut plan.num_steps, N, direction, flags)
    {
        return None;
    }

    plan.N = N;
    return Some(plan);
}

pub fn execute_function(func: u8, mut output: &mut [Complex32], input: &mut [Complex32], twiddles: &[Complex32], 
    twiddle_offset: u32, p:u32, samples: u32) {

        match func {
            0 => {
                mufft_forward_radix8_p1(&mut output, input, twiddles, twiddle_offset, p, samples);
            },
            1 =>  println!("mufft_forward_radix4_p1"),
            2 =>  println!("mufft_radix2_p1"),
            3 =>  println!("mufft_forward_half_radix8_p1"),
            4 =>  println!("mufft_forward_half_radix4_p1"),
            5 =>  println!("mufft_radix2_half_p1"),
            6 =>  println!("mufft_forward_radix2_p2"),
            7 =>  println!("mufft_inverse_radix8_p1"),
            8 =>  println!("mufft_inverse_radix4_p1"),
            9 =>  println!("mufft_inverse_radix2_p2"),
            10 => {
                mufft_radix8_generic(&mut output, input, twiddles, twiddle_offset, p, samples);
            },
            11 =>  {
                mufft_radix4_generic(&mut output, input, twiddles, twiddle_offset, p, samples);
            }
            12 => {
                mufft_radix2_generic(&mut output, input, twiddles, twiddle_offset, p, samples);
            },
            _ =>  println!("Not a valid function!!!"),
        }


}

pub fn mufft_execute_plan_1d(plan:  &mufft_plan_1d, output: &mut [Complex32], input: &mut [Complex32]) {
    let pt = &plan.twiddles;
    let N = plan.N;
    let steps: u32 = plan.num_steps;


    //TODO: 1 step FFT not working
    let first_step = &plan.steps[0]; 
    execute_function (first_step.func, output, input, &pt, 0, 1, N);
    let mut step_no = 0;
    
    for i in 1..steps
    {
        let step = &plan.steps[i as usize];
        if i%2 == 1 {
            execute_function(step.func, input, output, &pt, step.twiddle_offset, step.p, N);
        }
        else {
            execute_function(step.func, output, input, &pt, step.twiddle_offset, step.p, N);
        }
        step_no = i;

    }
    //Output is stored in input vector
    if step_no % 2 != 0 {
        let mut i = 0;
        while i < N {
            unsafe {
                let w = _mm256_load_ps(&input[i as usize].re as *const f32);
                _mm256_store_ps(&mut output[i as usize].re as *mut f32, w);
            }
            i += VSIZE;
        } 
    }
}
