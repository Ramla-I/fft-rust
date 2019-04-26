use num_complex::*;
// use alloc::vec::Vec;
use std::mem::transmute;

const M_SQRT1_2: f32 = 0.707106781186547524401;

cfg_if! { if #[cfg(target_feature = "avx")] {
    use std::arch::x86_64::*;

    const VSIZE: u32 = 4; // Complex numbers per vector

    // #[cfg(target_feature = "sse2")]
    // macro_rules! permute_ps {
    //     ($a: expr, $x: expr) => {
    //         _mm256_permute_ps($a, $x)
    //     };
    // }

    // #[cfg(target_feature = "sse2")]
    macro_rules! moveldup_ps {
        ($x: expr) => {
            _mm256_moveldup_ps($x)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! movehdup_ps {
        ($x: expr) => {
            _mm256_movehdup_ps($x)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! xor_ps {
        ($a: expr, $b: expr) => {
            _mm256_xor_ps($a, $b);
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! add_ps {
        ($a: expr, $b: expr) => {
            _mm256_add_ps($a, $b)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! sub_ps {
        ($a: expr, $b: expr) => {
            _mm256_sub_ps($a, $b)
        };
    }

    // #[cfg(target_feature = "sse2")]
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

    macro_rules! loadu_ps {
        ($a: expr) => {
            _mm256_loadu_ps($a)
        };
    }

    macro_rules! storeu_ps {
        ($a: expr, $b: expr) => {
            _mm256_storeu_ps($a, $b)
        };
    }

    macro_rules! storeu_ps {
        ($a: expr, $b: expr) => {
            _mm256_storeu_ps($a, $b)
        };
    }
    // #[cfg(target_feature = "sse2")]
    macro_rules! splat_const_complex {
        ($real: expr, $imag: expr) => {
            _mm256_set_ps($imag, $real, $imag, $real, $imag, $real, $imag, $real)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! unpacklo_pd {
        ($a: expr, $b: expr) => {
            _mm256_unpacklo_pd($a, $b) 
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! unpackhi_pd {
        ($a: expr, $b: expr) => {
            _mm256_unpackhi_pd($a, $b)
        };
    }



    // #[cfg(target_feature = "sse2")]
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

            // debug!("r3: {} {} {} {}", r3.extract(0) as f32, r3.extract(1) as f32, r3.extract(2) as f32, r3.extract(3) as f32);
            // debug!("r1: {} {} {} {}", r1.extract(0), r1.extract(1), r1.extract(2), r1.extract(3));
            // debug!("R0: {} {} {} {}", R0.extract(0), R0.extract(1), R0.extract(2), R0.extract(3));
            // debug!("r2: {} {} {} {}", r2.extract(0), r2.extract(1), r2.extract(2), r2.extract(3));
            // debug!("R1: {} {} {} {}", R1.extract(0), R1.extract(1), R1.extract(2), R1.extract(3));
            // debug!("");

            addsub_ps!(R0, R1)
        }
    }

    // pub fn add_avx (a: __m256, b: __m256) -> __m256 {
    //     unsafe{ _mm256_add_ps(a,b) }
    // }

    // #[cfg(target_feature = "sse2")]
    pub fn mufft_radix2_generic (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_radix2_generic avx");

        let half_samples: u32 = samples >> 1;
        let mut i = 0;

        // let twiddles_w_offset = &twiddles[twiddle_offset as usize].re as *const f32;

        while i < half_samples
        {
            let k: u32 = i & (p - 1);

            /* let twiddle_vector;
            unsafe {
                twiddle_vector = twiddles_w_offset.offset((k*2) as isize);// k*2 because complex numbers
            } 
            let input_vector = &input[i as usize].re as *const f32; */

            let (a, w, mut b);
            // let w;
            // let mut b;

            unsafe {
                w = loadu_ps!(&twiddles[(twiddle_offset -1 + k) as usize].re as *const f32);
                a = loadu_ps!(&input[i as usize].re as *const f32);
                b = loadu_ps!(&input[(i+half_samples) as usize].re as *const f32);
            

                b = cmul_ps(b, w);

                let r0 = add_ps!(a, b);
                let r1 = sub_ps!(a, b);

                let j: u32 = (i << 1) - k;
                //let output_vector = &mut output[j as usize].re as *mut f32;

            
                storeu_ps!(&mut output[j as usize].re as *mut f32, r0);
                storeu_ps!(&mut output[(j+p) as usize].re as *mut f32, r1); 
            }

            i += VSIZE;
        } 
    }

    pub fn mufft_radix4_generic (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        let quarter_samples: u32 = samples >> 2;
        let mut i = 0;

        while i < quarter_samples
        {
            let k: u32 = i & (p - 1);

            let (w, w0, w1, a, b, mut c, mut d);

            unsafe {
                w = loadu_ps!(&twiddles[(twiddle_offset -1 + k) as usize].re as *const f32);
                w0 = loadu_ps!(&twiddles[(twiddle_offset -1 + k + p) as usize].re as *const f32);
                w1 = loadu_ps!(&twiddles[(twiddle_offset -1 + k + 2*p) as usize].re as *const f32);

                a = loadu_ps!(&input[(i) as usize].re as *const f32);
                b = loadu_ps!(&input[(i + 1 * quarter_samples) as usize].re as *const f32);
                c = loadu_ps!(&input[(i + 2 * quarter_samples) as usize].re as *const f32);
                d = loadu_ps!(&input[(i + 3 * quarter_samples) as usize].re as *const f32);
            

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


                storeu_ps!(&mut output[(j + 0 * p) as usize].re as *mut f32, o0);
                storeu_ps!(&mut output[(j + 1 * p) as usize].re as *mut f32, o2);
                storeu_ps!(&mut output[(j + 2 * p) as usize].re as *mut f32, o1);
                storeu_ps!(&mut output[(j + 3 * p) as usize].re as *mut f32, o3);
            }

            i += VSIZE;
        }
    }

    // #[cfg(target_feature = "sse2")]
    pub fn mufft_forward_radix8_p1 (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_forward_radix8_p1 avx");

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
            let mut a;
            let mut b;
            let mut c;
            let mut d;
            let mut e;
            let mut f;
            let mut g;
            let mut h;

            // radix8_loadu_first_butterfly(i, octa_samples, &input, &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, &mut a, &mut b, &mut c, &mut d, &mut e ,&mut f, &mut g, &mut h); 
            
            /*************/
            // debug!("In radix8_loadu_first_butterfly");
            unsafe {
                a = loadu_ps!(&input[(i) as usize].re as *const f32); 
                b = loadu_ps!(&input[(i + 1 * octa_samples) as usize].re as *const f32); 
                c = loadu_ps!(&input[(i + 2 * octa_samples) as usize].re as *const f32); 
                d = loadu_ps!(&input[(i + 3 * octa_samples) as usize].re as *const f32); 
                e = loadu_ps!(&input[(i + 4 * octa_samples) as usize].re as *const f32); 
                f = loadu_ps!(&input[(i + 5 * octa_samples) as usize].re as *const f32); 
                g = loadu_ps!(&input[(i + 6 * octa_samples) as usize].re as *const f32); 
                h = loadu_ps!(&input[(i + 7 * octa_samples) as usize].re as *const f32); 
            

                // debug!("Loading r");
                let mut r0 = add_ps!(a, e); 
                let mut r1 = sub_ps!(a, e); 
                let mut r2 = add_ps!(b, f); 
                let mut r3 = sub_ps!(b, f); 
                let mut r4 = add_ps!(c, g); 
                let mut r5 = sub_ps!(c, g); 
                let mut r6 = add_ps!(d, h); 
                let mut r7 = sub_ps!(d, h);

                /***************/
                
                r5 = xor_ps!(_mm256_permute_ps(r5, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); 
                r7 = xor_ps!(_mm256_permute_ps(r7, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); 
        
                a = add_ps!(r0, r4); 
                b = add_ps!(r1, r5); 
                c = sub_ps!(r0, r4); 
                d = sub_ps!(r1, r5); 
                e = add_ps!(r2, r6); 
                f = add_ps!(r3, r7); 
                g = sub_ps!(r2, r6); 
                h = sub_ps!(r3, r7); 
        
                f = cmul_ps(f, w_f); 
                g = xor_ps!(_mm256_permute_ps(g, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); 
                h = cmul_ps(h, w_h); 

                // debug!("Loading o");
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
        
                // radix8_p1_end(&mut o0, &mut o1, &mut o2, &mut o3, &mut o4, &mut o5, &mut o6, &mut o7, &o0o1_lo, &o0o1_hi, &o2o3_lo, &o2o3_hi, &o4o5_lo, &o4o5_hi, &o6o7_lo, &o6o7_hi);

                /**************/

                // debug!("In radix8_p1_end");
            
                o0 = _mm256_permute2f128_ps(transmute(o0o1_lo), transmute(o2o3_lo), (2<<4) | (0<<0));
                o1 = _mm256_permute2f128_ps(transmute(o4o5_lo), transmute(o6o7_lo), (2<<4) | (0<<0));
                o2 = _mm256_permute2f128_ps(transmute(o0o1_hi), transmute(o2o3_hi), (2<<4) | (0<<0));
                o3 = _mm256_permute2f128_ps(transmute(o4o5_hi), transmute(o6o7_hi), (2<<4) | (0<<0));
                o4 = _mm256_permute2f128_ps(transmute(o0o1_lo), transmute(o2o3_lo), (3<<4) | (1<<0));
                o5 = _mm256_permute2f128_ps(transmute(o4o5_lo), transmute(o6o7_lo), (3<<4) | (1<<0));
                o6 = _mm256_permute2f128_ps(transmute(o0o1_hi), transmute(o2o3_hi), (3<<4) | (1<<0));
                o7 = _mm256_permute2f128_ps(transmute(o4o5_hi), transmute(o6o7_hi), (3<<4) | (1<<0));

                // o1 = transmute(o2o3_lo); 
                // o2 = transmute(o4o5_lo); 
                // o3 = transmute(o6o7_lo); 
                // o4 = transmute(o0o1_hi); 
                // o5 = transmute(o2o3_hi); 
                // o6 = transmute(o4o5_hi); 
                // o7 = transmute(o6o7_hi);
                
                /**************/

                let j: u32 = i << 3; 

                // debug!("Storing output");
                // debug!("Length of output: {}", output.len());
        
                
                storeu_ps!(&mut output[(j + 0 * VSIZE) as usize].re as *mut f32, o0); 
                storeu_ps!(&mut output[(j + 1 * VSIZE) as usize].re as *mut f32, o1); 
                storeu_ps!(&mut output[(j + 2 * VSIZE) as usize].re as *mut f32, o2); 
                storeu_ps!(&mut output[(j + 3 * VSIZE) as usize].re as *mut f32, o3); 
                storeu_ps!(&mut output[(j + 4 * VSIZE) as usize].re as *mut f32, o4); 
                storeu_ps!(&mut output[(j + 5 * VSIZE) as usize].re as *mut f32, o5); 
                storeu_ps!(&mut output[(j + 6 * VSIZE) as usize].re as *mut f32, o6); 
                storeu_ps!(&mut output[(j + 7 * VSIZE) as usize].re as *mut f32, o7);
            } 

            i += VSIZE
        } 

    }

    // #[cfg(target_feature = "sse2")]
    pub fn mufft_radix8_generic (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_radix8_generic");
        let octa_samples: u32 = samples >> 3;
        let mut i = 0;

        // let twiddles_w_offset = &twiddles[twiddle_offset as usize].re as *const f32;
        while i < octa_samples
        {
            // debug!("Round: {}", i);
            let k: u32 = i & (p - 1);
            //debug!("k: {}", k);
            // let twiddle_vector;
            // unsafe {
            //     twiddle_vector = twiddles_w_offset.offset((k*2) as isize);
            // } 
            // let input_vector = &input[i as usize].re as *const f32;
            
            let w;
            let mut a;
            let mut b;
            let mut c;
            let mut d;
            let mut e;
            let mut f;
            let mut g;
            let mut h;

            unsafe { 
                w = loadu_ps!(&twiddles[(twiddle_offset+k-1) as usize].re as *const f32); 
                a = loadu_ps!(&input[i as usize].re as *const f32);
                b = loadu_ps!(&input[(i + octa_samples) as usize].re as *const f32); 
                c = loadu_ps!(&input[(i + 2*octa_samples) as usize].re as *const f32);
                d = loadu_ps!(&input[(i + 3*octa_samples) as usize].re as *const f32);
                e = loadu_ps!(&input[(i + 4*octa_samples) as usize].re as *const f32);
                f = loadu_ps!(&input[(i + 5*octa_samples) as usize].re as *const f32);
                g = loadu_ps!(&input[(i + 6*octa_samples) as usize].re as *const f32);
                h = loadu_ps!(&input[(i + 7*octa_samples) as usize].re as *const f32);
            

                // debug!("{}", a.extract(0));
                // debug!("{}", b.extract(0));
                // debug!("{}", c.extract(0));
                // debug!("{}", d.extract(0));
                // debug!("{}", e.extract(0));
                // debug!("{}", f.extract(0));
                // debug!("{}", g.extract(0));
                // debug!("{}", h.extract(0));

                // debug!("w: {} {} {} {}", w.extract(0) as f32, w.extract(1) as f32, w.extract(2) as f32, w.extract(3) as f32);
                // debug!("e: {} {} {} {}", e.extract(0), e.extract(1), e.extract(2), e.extract(3));
                // debug!("f: {} {} {} {}", f.extract(0), f.extract(1), f.extract(2), f.extract(3));
                // debug!("g: {} {} {} {}", g.extract(0), g.extract(1), g.extract(2), g.extract(3));
                // debug!("h: {} {} {} {}", h.extract(0), h.extract(1), h.extract(2), h.extract(3));
                
                e = cmul_ps(e, w);
                f = cmul_ps(f, w);
                g = cmul_ps(g, w);
                h = cmul_ps(h, w);

                // debug!("w: {} {} {} {}", w.extract(0) as f32, w.extract(1) as f32, w.extract(2) as f32, w.extract(3) as f32);
                // debug!("e: {} {} {} {}", e.extract(0), e.extract(1), e.extract(2), e.extract(3));
                // debug!("f: {} {} {} {}", f.extract(0), f.extract(1), f.extract(2), f.extract(3));
                // debug!("g: {} {} {} {}", g.extract(0), g.extract(1), g.extract(2), g.extract(3));
                // debug!("h: {} {} {} {}", h.extract(0), h.extract(1), h.extract(2), h.extract(3));

                let r0 = add_ps!(a, e);
                let r1 = sub_ps!(a, e);
                let r2 = add_ps!(b, f);
                let r3 = sub_ps!(b, f);
                let mut r4 = add_ps!(c, g);
                let mut r5 = sub_ps!(c, g);
                let mut r6 = add_ps!(d, h);
                let mut r7 = sub_ps!(d, h);

                let (w0, w1);
                // let twiddle_vector_1 = &twiddles[(p + k) as usize].re as *const f32;
                // let twiddle_vector_2 = &twiddles[(2 * p + k) as usize].re as *const f32;
            
                w0 = loadu_ps!(&twiddles[(twiddle_offset + p + k-1) as usize].re as *const f32);
                w1 = loadu_ps!(&twiddles[(twiddle_offset + (2 * p + k-1)) as usize].re as *const f32);
            
                
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
                // let twiddle_vector_3 = &twiddles[(3 * p + k) as usize].re as *const f32;
                // let twiddle_vector_4 = &twiddles[(3 * p + k + p) as usize].re as *const f32;
                // let twiddle_vector_5 = &twiddles[(3 * p + k + 2 * p) as usize].re as *const f32;
                // let twiddle_vector_6 = &twiddles[(3 * p + k + 3 * p) as usize].re as *const f32;
            
                we = loadu_ps!(&twiddles[(twiddle_offset -1 + (3 * p + k)) as usize].re as *const f32);
                wf = loadu_ps!(&twiddles[(twiddle_offset -1 + (3 * p + k + p)) as usize].re as *const f32);
                wg = loadu_ps!(&twiddles[(twiddle_offset -1 + (3 * p + k + 2 * p)) as usize].re as *const f32);
                wh = loadu_ps!(&twiddles[(twiddle_offset -1 + (3 * p + k + 3 * p)) as usize].re as *const f32);
            
            
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
                // let output_vector = &mut output[j as usize].re as *mut f32;
            
                storeu_ps!(&mut output[(j + 0 * p) as usize].re as *mut f32, o0);
                storeu_ps!(&mut output[(j + 1 * p) as usize].re as *mut f32, o1);
                storeu_ps!(&mut output[(j + 2 * p) as usize].re as *mut f32, o2);
                storeu_ps!(&mut output[(j + 3 * p) as usize].re as *mut f32, o3);
                storeu_ps!(&mut output[(j + 4 * p) as usize].re as *mut f32, o4);
                storeu_ps!(&mut output[(j + 5 * p) as usize].re as *mut f32, o5);
                storeu_ps!(&mut output[(j + 6 * p) as usize].re as *mut f32, o6);
                storeu_ps!(&mut output[(j + 7 * p) as usize].re as *mut f32, o7);
            }

            i += VSIZE;
        }
    }
}

else {

    // #[cfg(target_feature="sse2")]
    use immintrin::xmmintrin::*;

    /* #[cfg(target_feature="sse2")]
    use immintrin::emmintrin::*; */

    // #[cfg(target_feature="sse2")]
    use immintrin::__m128;

    /* #[cfg(target_feature="sse2")]
    use immintrin::conversions::*; */

    const VSIZE: u32 = 2; // Complex numbers per vector

    // #[cfg(target_feature = "sse2")]
    macro_rules! permute_ps {
        ($a: expr, $x: expr) => {
            _mm_shuffle_ps($a, $a, $x)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! xor_ps {
        ($a: expr, $b: expr) => {
            _mm_xor_ps($a, $b);
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! add_ps {
        ($a: expr, $b: expr) => {
            _mm_add_ps($a, $b)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! sub_ps {
        ($a: expr, $b: expr) => {
            _mm_sub_ps($a, $b)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! mul_ps {
        ($a: expr, $b: expr) => {
        _mm_mul_ps($a, $b) 
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! moveldup_ps {
        ($x: expr) => {
            permute_ps!($x, _MM_SHUFFLE(2, 2, 0, 0))
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! movehdup_ps {
        ($x: expr) => {
            permute_ps!($x, _MM_SHUFFLE(3, 3, 1, 1))
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! splat_const_complex {
        ($real: expr, $imag: expr) => {
            _mm_set_ps($imag, $real, $imag, $real)
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! unpacklo_pd {
        ($a: expr, $b: expr) => {
            _mm_unpacklo_pd($a, $b) 
        };
    }

    // #[cfg(target_feature = "sse2")]
    macro_rules! unpackhi_pd {
        ($a: expr, $b: expr) => {
            _mm_unpackhi_pd($a, $b)
        };
    }

    // #[cfg(target_feature = "sse2")]
    #[inline]
    pub fn addsub_ps (a: __m128, b: __m128) -> __m128
    {
        let flip_signs: __m128 = splat_const_complex!(-0.0, 0.0);
        add_ps!(a, xor_ps!(b, flip_signs))
    }

    // #[cfg(target_feature = "sse2")]
    #[inline]
    pub fn cmul_ps(a: __m128, b: __m128) -> __m128
    {
        let c = _MM_SHUFFLE(2, 3, 0, 1);
        let r3 = permute_ps!(a,c);
        let r1 = moveldup_ps!(b);
        let R0 = mul_ps!(a, r1);
        let r2 = movehdup_ps!(b);
        let R1 = mul_ps!(r2, r3);

        // debug!("r3: {} {} {} {}", r3.extract(0) as f32, r3.extract(1) as f32, r3.extract(2) as f32, r3.extract(3) as f32);
        // debug!("r1: {} {} {} {}", r1.extract(0), r1.extract(1), r1.extract(2), r1.extract(3));
        // debug!("R0: {} {} {} {}", R0.extract(0), R0.extract(1), R0.extract(2), R0.extract(3));
        // debug!("r2: {} {} {} {}", r2.extract(0), r2.extract(1), r2.extract(2), r2.extract(3));
        // debug!("R1: {} {} {} {}", R1.extract(0), R1.extract(1), R1.extract(2), R1.extract(3));
        // debug!("");

        addsub_ps(R0, R1)
    }

    // #[cfg(target_feature = "sse2")]
    pub fn mufft_radix2_generic (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_radix2_generic");

        let half_samples: u32 = samples >> 1;
        let mut i = 0;

        // let twiddles_w_offset = &twiddles[twiddle_offset as usize].re as *const f32;

        while i < half_samples
        {
            let k: u32 = i & (p - 1);

            /* let twiddle_vector;
            unsafe {
                twiddle_vector = twiddles_w_offset.offset((k*2) as isize);// k*2 because complex numbers
            } 
            let input_vector = &input[i as usize].re as *const f32; */

            let (a, w, mut b);
            // let w;
            // let mut b;

            unsafe {
                w = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + k) as usize].re as *const f32);
                a = _mm_loadu_ps(&input[i as usize].re as *const f32);
                b = _mm_loadu_ps(&input[(i+half_samples) as usize].re as *const f32);
            }

            b = cmul_ps(b, w);

            let r0 = add_ps!(a, b);
            let r1 = sub_ps!(a, b);

            let j: u32 = (i << 1) - k;
            //let output_vector = &mut output[j as usize].re as *mut f32;

            unsafe {
                _mm_storeu_ps(&mut output[j as usize].re as *mut f32, r0);
                _mm_storeu_ps(&mut output[(j+p) as usize].re as *mut f32, r1); 
            }

            i += VSIZE;
        } 
    }

    pub fn mufft_radix4_generic (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_radix2_generic");
        let quarter_samples: u32 = samples >> 2;
        let mut i = 0;

        while i < quarter_samples
        {
            let k: u32 = i & (p - 1);

            let (w, w0, w1, a, b, mut c, mut d);

            unsafe {
                w = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + k) as usize].re as *const f32);
                w0 = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + k + p) as usize].re as *const f32);
                w1 = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + k + 2*p) as usize].re as *const f32);

                a = _mm_loadu_ps(&input[(i) as usize].re as *const f32);
                b = _mm_loadu_ps(&input[(i + 1 * quarter_samples) as usize].re as *const f32);
                c = _mm_loadu_ps(&input[(i + 2 * quarter_samples) as usize].re as *const f32);
                d = _mm_loadu_ps(&input[(i + 3 * quarter_samples) as usize].re as *const f32);
            }

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

            unsafe {
                _mm_storeu_ps(&mut output[(j + 0 * p) as usize].re as *mut f32, o0);
                _mm_storeu_ps(&mut output[(j + 1 * p) as usize].re as *mut f32, o2);
                _mm_storeu_ps(&mut output[(j + 2 * p) as usize].re as *mut f32, o1);
                _mm_storeu_ps(&mut output[(j + 3 * p) as usize].re as *mut f32, o3);
            }

            i += VSIZE;
        }
    }

    // #[cfg(target_feature = "sse2")]
    pub fn mufft_forward_radix8_p1 (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_forward_radix8_p1");

        let twiddle_r: f32 = 0.0;
        let twiddle_i: f32 = -0.0;
        let twiddle8: f32 = -M_SQRT1_2;
    
        let flip_signs = splat_const_complex!(twiddle_r, twiddle_i); 
        let w_f = splat_const_complex!(M_SQRT1_2, twiddle8); 
        let w_h = splat_const_complex!(-M_SQRT1_2, twiddle8); 
    
        let octa_samples: u32 = samples >> 3; 
        let mut i = 0;
        while i < octa_samples
        { 
            let mut a;
            let mut b;
            let mut c;
            let mut d;
            let mut e;
            let mut f;
            let mut g;
            let mut h;

            // radix8_loadu_first_butterfly(i, octa_samples, &input, &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, &mut a, &mut b, &mut c, &mut d, &mut e ,&mut f, &mut g, &mut h); 
            
            /*************/
            // debug!("In radix8_loadu_first_butterfly");
            unsafe {
                a = _mm_loadu_ps(&input[(i) as usize].re as *const f32); 
                b = _mm_loadu_ps(&input[(i + 1 * octa_samples) as usize].re as *const f32); 
                c = _mm_loadu_ps(&input[(i + 2 * octa_samples) as usize].re as *const f32); 
                d = _mm_loadu_ps(&input[(i + 3 * octa_samples) as usize].re as *const f32); 
                e = _mm_loadu_ps(&input[(i + 4 * octa_samples) as usize].re as *const f32); 
                f = _mm_loadu_ps(&input[(i + 5 * octa_samples) as usize].re as *const f32); 
                g = _mm_loadu_ps(&input[(i + 6 * octa_samples) as usize].re as *const f32); 
                h = _mm_loadu_ps(&input[(i + 7 * octa_samples) as usize].re as *const f32); 
            }

            // debug!("Loading r");
            let mut r0 = add_ps!(a, e); 
            let mut r1 = sub_ps!(a, e); 
            let mut r2 = add_ps!(b, f); 
            let mut r3 = sub_ps!(b, f); 
            let mut r4 = add_ps!(c, g); 
            let mut r5 = sub_ps!(c, g); 
            let mut r6 = add_ps!(d, h); 
            let mut r7 = sub_ps!(d, h);

            /***************/
            
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

            // debug!("Loading o");
            let mut o0 = add_ps!(a, e); 
            let mut o1 = add_ps!(b, f); 
            let mut o2 = add_ps!(c, g); 
            let mut o3 = add_ps!(d, h); 
            let mut o4 = sub_ps!(a, e); 
            let mut o5 = sub_ps!(b, f); 
            let mut o6 = sub_ps!(c, g); 
            let mut o7 = sub_ps!(d, h); 
    
            let o0o1_lo = unpacklo_pd!(o0, o1); 
            let o0o1_hi = unpackhi_pd!(o0, o1); 
            let o2o3_lo = unpacklo_pd!(o2, o3); 
            let o2o3_hi = unpackhi_pd!(o2, o3); 
            let o4o5_lo = unpacklo_pd!(o4, o5); 
            let o4o5_hi = unpackhi_pd!(o4, o5); 
            let o6o7_lo = unpacklo_pd!(o6, o7); 
            let o6o7_hi = unpackhi_pd!(o6, o7); 
    
            // radix8_p1_end(&mut o0, &mut o1, &mut o2, &mut o3, &mut o4, &mut o5, &mut o6, &mut o7, &o0o1_lo, &o0o1_hi, &o2o3_lo, &o2o3_hi, &o4o5_lo, &o4o5_hi, &o6o7_lo, &o6o7_hi);

            /**************/

            // debug!("In radix8_p1_end");
            o0 = o0o1_lo; 
            o1 = o2o3_lo; 
            o2 = o4o5_lo; 
            o3 = o6o7_lo; 
            o4 = o0o1_hi; 
            o5 = o2o3_hi; 
            o6 = o4o5_hi; 
            o7 = o6o7_hi;

            /**************/

            let j: u32 = i << 3; 

            // debug!("Storing output");
            // debug!("Length of output: {}", output.len());
            unsafe {
                
                _mm_storeu_ps(&mut output[(j + 0 * VSIZE) as usize].re as *mut f32, o0); 
                _mm_storeu_ps(&mut output[(j + 1 * VSIZE) as usize].re as *mut f32, o1); 
                _mm_storeu_ps(&mut output[(j + 2 * VSIZE) as usize].re as *mut f32, o2); 
                _mm_storeu_ps(&mut output[(j + 3 * VSIZE) as usize].re as *mut f32, o3); 
                _mm_storeu_ps(&mut output[(j + 4 * VSIZE) as usize].re as *mut f32, o4); 
                _mm_storeu_ps(&mut output[(j + 5 * VSIZE) as usize].re as *mut f32, o5); 
                _mm_storeu_ps(&mut output[(j + 6 * VSIZE) as usize].re as *mut f32, o6); 
                _mm_storeu_ps(&mut output[(j + 7 * VSIZE) as usize].re as *mut f32, o7);
            } 

            i += VSIZE
        } 

    }

    // #[cfg(target_feature = "sse2")]
    pub fn mufft_radix8_generic (output: &mut Vec<Complex32>, input: &Vec<Complex32>, twiddles: &Vec<Complex32>, twiddle_offset: u32, p: u32, samples: u32) {
        // debug!("In mufft_radix8_generic");
        let octa_samples: u32 = samples >> 3;
        let mut i = 0;

        // let twiddles_w_offset = &twiddles[twiddle_offset as usize].re as *const f32;
        while i < octa_samples
        {
            // debug!("Round: {} start", i);
            let k: u32 = i & (p - 1);
            //debug!("k: {}", k);
            // let twiddle_vector;
            // unsafe {
            //     twiddle_vector = twiddles_w_offset.offset((k*2) as isize);
            // } 
            // let input_vector = &input[i as usize].re as *const f32;
            
            let w;
            let mut a;
            let mut b;
            let mut c;
            let mut d;
            let mut e;
            let mut f;
            let mut g;
            let mut h;

            unsafe { 
                w = _mm_loadu_ps(&twiddles[(twiddle_offset+k-1) as usize].re as *const f32); 
                a = _mm_loadu_ps(&input[i as usize].re as *const f32);
                b = _mm_loadu_ps(&input[(i + octa_samples) as usize].re as *const f32); 
                c = _mm_loadu_ps(&input[(i + 2*octa_samples) as usize].re as *const f32);
                d = _mm_loadu_ps(&input[(i + 3*octa_samples) as usize].re as *const f32);
                e = _mm_loadu_ps(&input[(i + 4*octa_samples) as usize].re as *const f32);
                f = _mm_loadu_ps(&input[(i + 5*octa_samples) as usize].re as *const f32);
                g = _mm_loadu_ps(&input[(i + 6*octa_samples) as usize].re as *const f32);
                h = _mm_loadu_ps(&input[(i + 7*octa_samples) as usize].re as *const f32);
            }

            // debug!("{}", a.extract(0));
            // debug!("{}", b.extract(0));
            // debug!("{}", c.extract(0));
            // debug!("{}", d.extract(0));
            // debug!("{}", e.extract(0));
            // debug!("{}", f.extract(0));
            // debug!("{}", g.extract(0));
            // debug!("{}", h.extract(0));

            // debug!("w: {} {} {} {}", w.extract(0) as f32, w.extract(1) as f32, w.extract(2) as f32, w.extract(3) as f32);
            // debug!("e: {} {} {} {}", e.extract(0), e.extract(1), e.extract(2), e.extract(3));
            // debug!("f: {} {} {} {}", f.extract(0), f.extract(1), f.extract(2), f.extract(3));
            // debug!("g: {} {} {} {}", g.extract(0), g.extract(1), g.extract(2), g.extract(3));
            // debug!("h: {} {} {} {}", h.extract(0), h.extract(1), h.extract(2), h.extract(3));
            
            e = cmul_ps(e, w);
            f = cmul_ps(f, w);
            g = cmul_ps(g, w);
            h = cmul_ps(h, w);

            // debug!("w: {} {} {} {}", w.extract(0) as f32, w.extract(1) as f32, w.extract(2) as f32, w.extract(3) as f32);
            // debug!("e: {} {} {} {}", e.extract(0), e.extract(1), e.extract(2), e.extract(3));
            // debug!("f: {} {} {} {}", f.extract(0), f.extract(1), f.extract(2), f.extract(3));
            // debug!("g: {} {} {} {}", g.extract(0), g.extract(1), g.extract(2), g.extract(3));
            // debug!("h: {} {} {} {}", h.extract(0), h.extract(1), h.extract(2), h.extract(3));

            let r0 = add_ps!(a, e);
            let r1 = sub_ps!(a, e);
            let r2 = add_ps!(b, f);
            let r3 = sub_ps!(b, f);
            let mut r4 = add_ps!(c, g);
            let mut r5 = sub_ps!(c, g);
            let mut r6 = add_ps!(d, h);
            let mut r7 = sub_ps!(d, h);

            let (w0, w1);
            // let twiddle_vector_1 = &twiddles[(p + k) as usize].re as *const f32;
            // let twiddle_vector_2 = &twiddles[(2 * p + k) as usize].re as *const f32;
            unsafe{
                w0 = _mm_loadu_ps(&twiddles[(twiddle_offset + p + k-1) as usize].re as *const f32);
                w1 = _mm_loadu_ps(&twiddles[(twiddle_offset + (2 * p + k-1)) as usize].re as *const f32);
            }
            
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
            // let twiddle_vector_3 = &twiddles[(3 * p + k) as usize].re as *const f32;
            // let twiddle_vector_4 = &twiddles[(3 * p + k + p) as usize].re as *const f32;
            // let twiddle_vector_5 = &twiddles[(3 * p + k + 2 * p) as usize].re as *const f32;
            // let twiddle_vector_6 = &twiddles[(3 * p + k + 3 * p) as usize].re as *const f32;
            unsafe {
                we = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + (3 * p + k)) as usize].re as *const f32);
                wf = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + (3 * p + k + p)) as usize].re as *const f32);
                wg = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + (3 * p + k + 2 * p)) as usize].re as *const f32);
                wh = _mm_loadu_ps(&twiddles[(twiddle_offset -1 + (3 * p + k + 3 * p)) as usize].re as *const f32);
            }
            
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
            // let output_vector = &mut output[j as usize].re as *mut f32;
            unsafe {
                _mm_storeu_ps(&mut output[(j + 0 * p) as usize].re as *mut f32, o0);
                _mm_storeu_ps(&mut output[(j + 1 * p) as usize].re as *mut f32, o1);
                _mm_storeu_ps(&mut output[(j + 2 * p) as usize].re as *mut f32, o2);
                _mm_storeu_ps(&mut output[(j + 3 * p) as usize].re as *mut f32, o3);
                _mm_storeu_ps(&mut output[(j + 4 * p) as usize].re as *mut f32, o4);
                _mm_storeu_ps(&mut output[(j + 5 * p) as usize].re as *mut f32, o5);
                _mm_storeu_ps(&mut output[(j + 6 * p) as usize].re as *mut f32, o6);
                _mm_storeu_ps(&mut output[(j + 7 * p) as usize].re as *mut f32, o7);
            }

            i += VSIZE;
            // debug!("Round: {} end", i);
        }
    }
}
}