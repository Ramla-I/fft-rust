use num_complex::*;
// use std::vec::Vec;
use std::mem::transmute;
use std::arch::x86_64::*;

const M_SQRT1_2: f32 = 0.707106781186547524401;
pub const VSIZE: u32 = 4; // Complex numbers per vector

macro_rules! permute_ps {
    ($a: expr, $x: expr) => {
        _mm256_permute_ps($a, $x)
    };
}

macro_rules! moveldup_ps {
    ($x: expr) => {
        _mm256_moveldup_ps($x)
    };
}

macro_rules! movehdup_ps {
    ($x: expr) => {
        _mm256_movehdup_ps($x)
    };
}

macro_rules! xor_ps {
    ($a: expr, $b: expr) => {
        _mm256_xor_ps($a, $b);
    };
}

macro_rules! add_ps {
    ($a: expr, $b: expr) => {
        _mm256_add_ps($a, $b)
    };
}

macro_rules! sub_ps {
    ($a: expr, $b: expr) => {
        _mm256_sub_ps($a, $b)
    };
}

macro_rules! mul_ps {
    ($a: expr, $b: expr) => {
    _mm256_mul_ps($a, $b) 
    };
}

macro_rules! addsub_ps {
    ($a: expr, $b: expr) => {
        _mm256_addsub_ps($a, $b)
    };
}

macro_rules! loadu_ps {
    ($a: expr) => {
        _mm256_loadu_ps($a)
    };
}

#[macro_export]
macro_rules! load_ps {
    ($a: expr) => {
        _mm256_load_ps($a)
    };
}

#[macro_export]
macro_rules! store_ps {
    ($a: expr, $b: expr) => {
        _mm256_store_ps($a, $b)
    };
}

macro_rules! storeu_ps {
    ($a: expr, $b: expr) => {
        _mm256_storeu_ps($a, $b)
    };
}

macro_rules! splat_const_complex {
    ($real: expr, $imag: expr) => {
        _mm256_set_ps($imag, $real, $imag, $real, $imag, $real, $imag, $real)
    };
}

macro_rules! unpacklo_pd {
    ($a: expr, $b: expr) => {
        _mm256_unpacklo_pd($a, $b) 
    };
}

macro_rules! unpackhi_pd {
    ($a: expr, $b: expr) => {
        _mm256_unpackhi_pd($a, $b)
    };
}

#[inline]
pub fn cmul_ps(a: __m256, b: __m256) -> __m256
{
    unsafe{
        const c: i32 = _MM_SHUFFLE(2, 3, 0, 1);
        let r3 = _mm256_permute_ps(a,c);
        let r1 = moveldup_ps!(b);
        let R0 = mul_ps!(a, r1);
        let r2 = movehdup_ps!(b);
        let R1 = mul_ps!(r2, r3);

        addsub_ps!(R0, R1)
    }
}


pub fn mufft_radix2_generic (output: &mut [Complex32], input: &[Complex32], twiddles: &[Complex32], twiddle_offset: u32, p: u32, samples: u32) {

    let half_samples: u32 = samples >> 1;
    let mut i = 0;

    while i < half_samples
    {
        let k: u32 = i & (p - 1);

        let (a, w, mut b);

        unsafe {
            w = load_ps!(&twiddles[(twiddle_offset + k) as usize].re as *const f32);
            a = load_ps!(&input[i as usize].re as *const f32);
            b = load_ps!(&input[(i+half_samples) as usize].re as *const f32);
        

            b = cmul_ps(b, w);

            let r0 = add_ps!(a, b);
            let r1 = sub_ps!(a, b);

            let j: u32 = (i << 1) - k;
            //let output_vector = &mut output[j as usize].re as *mut f32;

        
            store_ps!(&mut output[j as usize].re as *mut f32, r0);
            store_ps!(&mut output[(j+p) as usize].re as *mut f32, r1); 
        }

        i += VSIZE;
    } 
}

pub fn mufft_radix4_generic (output: &mut [Complex32], input: &[Complex32], twiddles: &[Complex32], twiddle_offset: u32, p: u32, samples: u32) {
    let quarter_samples: u32 = samples >> 2;
    let mut i = 0;

    while i < quarter_samples
    {
        let k: u32 = i & (p - 1);

        let (w, w0, w1, a, b, mut c, mut d);

        unsafe {
            w = load_ps!(&twiddles[(twiddle_offset + k) as usize].re as *const f32);
            w0 = load_ps!(&twiddles[(twiddle_offset + k + p) as usize].re as *const f32);
            w1 = load_ps!(&twiddles[(twiddle_offset + k + 2*p) as usize].re as *const f32);

            a = load_ps!(&input[(i) as usize].re as *const f32);
            b = load_ps!(&input[(i + 1 * quarter_samples) as usize].re as *const f32);
            c = load_ps!(&input[(i + 2 * quarter_samples) as usize].re as *const f32);
            d = load_ps!(&input[(i + 3 * quarter_samples) as usize].re as *const f32);
        

            c = cmul_ps(c, w);
            d = cmul_ps(d, w);

            let r0 = add_ps!(a, c);
            let r1 = sub_ps!(a, c);
            let mut r2 = add_ps!(b, d);
            let mut r3 = sub_ps!(b, d);

            r2 = cmul_ps(r2, w0);
            r3 = cmul_ps(r3, w1);

            let o0 = add_ps!(r0, r2);
            let o1 = sub_ps!(r0, r2);
            let o2 = add_ps!(r1, r3);
            let o3 = sub_ps!(r1, r3);

            let j: u32 = ((i - k) << 2) + k;


            store_ps!(&mut output[(j + 0 * p) as usize].re as *mut f32, o0);
            store_ps!(&mut output[(j + 1 * p) as usize].re as *mut f32, o2);
            store_ps!(&mut output[(j + 2 * p) as usize].re as *mut f32, o1);
            store_ps!(&mut output[(j + 3 * p) as usize].re as *mut f32, o3);
        }

        i += VSIZE;
    }
}

// #[cfg(target_feature = "sse2")]
pub fn mufft_forward_radix8_p1 (output: &mut [Complex32], input: &[Complex32], twiddles: &[Complex32], twiddle_offset: u32, p: u32, samples: u32) {

    let twiddle_r: f32 = 0.0;
    let twiddle_i: f32 = -0.0;
    let twiddle8: f32 = -M_SQRT1_2;

    let (flip_signs, w_f, w_h);
    unsafe {
        flip_signs = splat_const_complex!(twiddle_r, twiddle_i); 
        w_f = splat_const_complex!(M_SQRT1_2, twiddle8); 
        w_h = splat_const_complex!(-M_SQRT1_2, twiddle8); 
    }

    let octa_samples: u32 = samples >> 3; 
    let mut i = 0;
    while i < octa_samples
    { 
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h);

        unsafe {
            a = load_ps!(&input[(i) as usize].re as *const f32); 
            b = load_ps!(&input[(i + 1 * octa_samples) as usize].re as *const f32); 
            c = load_ps!(&input[(i + 2 * octa_samples) as usize].re as *const f32); 
            d = load_ps!(&input[(i + 3 * octa_samples) as usize].re as *const f32); 
            e = load_ps!(&input[(i + 4 * octa_samples) as usize].re as *const f32); 
            f = load_ps!(&input[(i + 5 * octa_samples) as usize].re as *const f32); 
            g = load_ps!(&input[(i + 6 * octa_samples) as usize].re as *const f32); 
            h = load_ps!(&input[(i + 7 * octa_samples) as usize].re as *const f32); 
        
            let r0 = add_ps!(a, e); 
            let r1 = sub_ps!(a, e); 
            let r2 = add_ps!(b, f); 
            let r3 = sub_ps!(b, f); 
            let r4 = add_ps!(c, g); 
            let mut r5 = sub_ps!(c, g); 
            let r6 = add_ps!(d, h); 
            let mut r7 = sub_ps!(d, h);
          
            r5 = xor_ps!(permute_ps!(r5, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); 
            r7 = xor_ps!(permute_ps!(r7, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); 
    
            a = add_ps!(r0, r4); 
            b = add_ps!(r1, r5); 
            c = sub_ps!(r0, r4); 
            d = sub_ps!(r1, r5); 
            e = add_ps!(r2, r6); 
            f = add_ps!(r3, r7); 
            g = sub_ps!(r2, r6); 
            h = sub_ps!(r3, r7); 
    
            f = cmul_ps(f, w_f); 
            g = xor_ps!(permute_ps!(g, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); 
            h = cmul_ps(h, w_h); 

            let mut o0 = add_ps!(a, e); 
            let mut o1 = add_ps!(b, f);
            let mut o2 = add_ps!(c, g); 
            let mut o3 = add_ps!(d, h); 
            let mut o4 = sub_ps!(a, e); 
            let mut o5 = sub_ps!(b, f); 
            let mut o6 = sub_ps!(c, g); 
            let mut o7 = sub_ps!(d, h); 
            
            let o0d = transmute(o0);
            let o1d = transmute(o1);
            let o2d = transmute(o2);
            let o3d = transmute(o3);
            let o4d = transmute(o4);
            let o5d = transmute(o5);
            let o6d = transmute(o6);
            let o7d = transmute(o7);


            let o0o1_lo = unpacklo_pd!(o0d, o1d); 
            let o0o1_hi = unpackhi_pd!(o0d, o1d); 
            let o2o3_lo = unpacklo_pd!(o2d, o3d); 
            let o2o3_hi = unpackhi_pd!(o2d, o3d); 
            let o4o5_lo = unpacklo_pd!(o4d, o5d); 
            let o4o5_hi = unpackhi_pd!(o4d, o5d); 
            let o6o7_lo = unpacklo_pd!(o6d, o7d); 
            let o6o7_hi = unpackhi_pd!(o6d, o7d); 

            o0 = _mm256_permute2f128_ps(transmute(o0o1_lo), transmute(o2o3_lo), (2<<4) | (0<<0));
            o1 = _mm256_permute2f128_ps(transmute(o4o5_lo), transmute(o6o7_lo), (2<<4) | (0<<0));
            o2 = _mm256_permute2f128_ps(transmute(o0o1_hi), transmute(o2o3_hi), (2<<4) | (0<<0));
            o3 = _mm256_permute2f128_ps(transmute(o4o5_hi), transmute(o6o7_hi), (2<<4) | (0<<0));
            o4 = _mm256_permute2f128_ps(transmute(o0o1_lo), transmute(o2o3_lo), (3<<4) | (1<<0));
            o5 = _mm256_permute2f128_ps(transmute(o4o5_lo), transmute(o6o7_lo), (3<<4) | (1<<0));
            o6 = _mm256_permute2f128_ps(transmute(o0o1_hi), transmute(o2o3_hi), (3<<4) | (1<<0));
            o7 = _mm256_permute2f128_ps(transmute(o4o5_hi), transmute(o6o7_hi), (3<<4) | (1<<0));

            let j: u32 = i << 3; 
            
            store_ps!(&mut output[(j + 0 * VSIZE) as usize].re as *mut f32, o0); 
            store_ps!(&mut output[(j + 1 * VSIZE) as usize].re as *mut f32, o1); 
            store_ps!(&mut output[(j + 2 * VSIZE) as usize].re as *mut f32, o2); 
            store_ps!(&mut output[(j + 3 * VSIZE) as usize].re as *mut f32, o3); 
            store_ps!(&mut output[(j + 4 * VSIZE) as usize].re as *mut f32, o4); 
            store_ps!(&mut output[(j + 5 * VSIZE) as usize].re as *mut f32, o5); 
            store_ps!(&mut output[(j + 6 * VSIZE) as usize].re as *mut f32, o6); 
            store_ps!(&mut output[(j + 7 * VSIZE) as usize].re as *mut f32, o7);
        } 

        i += VSIZE
    } 

}

pub fn mufft_radix8_generic (output: &mut [Complex32], input: &[Complex32], twiddles: &[Complex32], twiddle_offset: u32, p: u32, samples: u32) {

    let octa_samples: u32 = samples >> 3;
    let mut i = 0;

    while i < octa_samples
    {
        let k: u32 = i & (p - 1);

        let (w, mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h);

        unsafe { 
            w = load_ps!(&twiddles[(twiddle_offset+k) as usize].re as *const f32); 
            a = load_ps!(&input[i as usize].re as *const f32);
            b = load_ps!(&input[(i + octa_samples) as usize].re as *const f32); 
            c = load_ps!(&input[(i + 2*octa_samples) as usize].re as *const f32);
            d = load_ps!(&input[(i + 3*octa_samples) as usize].re as *const f32);
            e = load_ps!(&input[(i + 4*octa_samples) as usize].re as *const f32);
            f = load_ps!(&input[(i + 5*octa_samples) as usize].re as *const f32);
            g = load_ps!(&input[(i + 6*octa_samples) as usize].re as *const f32);
            h = load_ps!(&input[(i + 7*octa_samples) as usize].re as *const f32);

            e = cmul_ps(e, w);
            f = cmul_ps(f, w);
            g = cmul_ps(g, w);
            h = cmul_ps(h, w);

            let r0 = add_ps!(a, e);
            let r1 = sub_ps!(a, e);
            let r2 = add_ps!(b, f);
            let r3 = sub_ps!(b, f);
            let mut r4 = add_ps!(c, g);
            let mut r5 = sub_ps!(c, g);
            let mut r6 = add_ps!(d, h);
            let mut r7 = sub_ps!(d, h);

            let (w0, w1);
        
            w0 = load_ps!(&twiddles[(twiddle_offset + p + k) as usize].re as *const f32);
            w1 = load_ps!(&twiddles[(twiddle_offset + (2 * p + k)) as usize].re as *const f32);
        
            
            r4 = cmul_ps(r4, w0);
            r5 = cmul_ps(r5, w1);
            r6 = cmul_ps(r6, w0);
            r7 = cmul_ps(r7, w1);

            a = add_ps!(r0, r4);
            b = add_ps!(r1, r5);
            c = sub_ps!(r0, r4);
            d = sub_ps!(r1, r5);
            e = add_ps!(r2, r6);
            f = add_ps!(r3, r7);
            g = sub_ps!(r2, r6);
            h = sub_ps!(r3, r7);

            let (we,wf,wg,wh);
        
            we = load_ps!(&twiddles[(twiddle_offset + (3 * p + k)) as usize].re as *const f32);
            wf = load_ps!(&twiddles[(twiddle_offset + (3 * p + k + p)) as usize].re as *const f32);
            wg = load_ps!(&twiddles[(twiddle_offset + (3 * p + k + 2 * p)) as usize].re as *const f32);
            wh = load_ps!(&twiddles[(twiddle_offset + (3 * p + k + 3 * p)) as usize].re as *const f32);
        
            e = cmul_ps(e, we);
            f = cmul_ps(f, wf);
            g = cmul_ps(g, wg);
            h = cmul_ps(h, wh);

            let o0 = add_ps!(a, e);
            let o1 = add_ps!(b, f);
            let o2 = add_ps!(c, g);
            let o3 = add_ps!(d, h);
            let o4 = sub_ps!(a, e);
            let o5 = sub_ps!(b, f);
            let o6 = sub_ps!(c, g);
            let o7 = sub_ps!(d, h);

            let j = ((i - k) << 3) + k;
        
            store_ps!(&mut output[(j + 0 * p) as usize].re as *mut f32, o0);
            store_ps!(&mut output[(j + 1 * p) as usize].re as *mut f32, o1);
            store_ps!(&mut output[(j + 2 * p) as usize].re as *mut f32, o2);
            store_ps!(&mut output[(j + 3 * p) as usize].re as *mut f32, o3);
            store_ps!(&mut output[(j + 4 * p) as usize].re as *mut f32, o4);
            store_ps!(&mut output[(j + 5 * p) as usize].re as *mut f32, o5);
            store_ps!(&mut output[(j + 6 * p) as usize].re as *mut f32, o6);
            store_ps!(&mut output[(j + 7 * p) as usize].re as *mut f32, o7);
        }

        i += VSIZE;
    }
}


