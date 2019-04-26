#![feature(stdsimd)]
#[macro_use] extern crate log;
#[macro_use] extern crate cfg_if;

extern crate num_complex;
extern crate immintrin;
extern crate libm;
extern crate rand;

use std::mem;
use num_complex::Complex32;
// use alloc::vec::Vec;

pub mod test;
pub mod fft;
pub mod kernel;

use test::*;

// Raising alignment
#[repr(align(16))]
struct Align16(u64,u64);

#[repr(align(32))]
struct Align32(u64,u64,u64,u64);

#[repr(align(64))]
struct Align64(u64,u64,u64,u64,u64,u64,u64,u64);

///function to create a Vec aligned at 16 bytes
pub fn allocate_aligned_vec_16(size: usize, vec: &mut Vec<Complex32>) {

    let buffer : Vec<Align16> = Vec::with_capacity(size/2);
    
    let buffer_ptr = buffer.as_slice().as_ptr() as usize;

    // debug!("Aligned pointer: {:#X}", buffer_ptr);

    let ptr = buffer_ptr as *mut Complex32;
    
    mem::forget(buffer);

    *vec = unsafe {Vec::from_raw_parts(ptr, size, size)};

    for i in vec {
        i.re = 0.0;
        i.im = 0.0;
    }
}

///function to create a Vec aligned at 16 bytes
pub fn allocate_aligned_vec_32(size: usize, vec: &mut Vec<Complex32>) {

    let buffer : Vec<Align32> = Vec::with_capacity(size/4);
    
    let buffer_ptr = buffer.as_slice().as_ptr() as usize;

    // debug!("Aligned pointer: {:#X}", buffer_ptr);

    let ptr = buffer_ptr as *mut Complex32;
    
    mem::forget(buffer);

    *vec = unsafe {Vec::from_raw_parts(ptr, size, size)};

    for i in vec {
        i.re = 0.0;
        i.im = 0.0;
    }
}

///function to create a Vec aligned at 16 bytes
pub fn allocate_aligned_vec_64(size: usize, vec: &mut Vec<Complex32>) {

    let buffer : Vec<Align64> = Vec::with_capacity(size/8);
    
    let buffer_ptr = buffer.as_slice().as_ptr() as usize;

    // debug!("Aligned pointer: {:#X}", buffer_ptr);

    let ptr = buffer_ptr as *mut Complex32;
    
    mem::forget(buffer);

    *vec = unsafe {Vec::from_raw_parts(ptr, size, size)};

    for i in vec {
        i.re = 0.0;
        i.im = 0.0;
    }
}

fn main() {
    // testing_fft(2048);
    bm_fft(2048);

}


