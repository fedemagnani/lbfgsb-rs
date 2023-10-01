use std::ops::{Sub,Mul};
use crate::{
    common::*,
    lbfgsb::LBFGSB
};

pub fn objective_function(x:&Vec<double>, f:&mut double){
    let n = x.len();
    let mut d__1 = x[0].sub(1.0);
    *f = (d__1 * d__1).mul(0.25);
    let i__1 = n;
    
    for i__ in 2..=i__1 {
        /* Computing 2nd power */
        let d__2 = x[i__ - 2];
        /* Computing 2nd power */
        d__1 = x[i__ - 1] - d__2 * d__2;
        *f += d__1 * d__1;
    }
    *f = f.mul(4.);
}

pub fn gradient_function(x:&Vec<double>, g:&mut Vec<double>){
        let n = x.len();
        let mut d__1 = x[0];
        let mut t1 = x[1] - d__1 * d__1;
        g[0] = (x[0] - 1.) * 2. - x[0] * 16. * t1;
        let i__1 = n - 1;
        for i__ in 2..=i__1 {
            let t2 = t1;
            /* Computing 2nd power */
            d__1 = x[i__ - 1];
            t1 = x[i__] - d__1 * d__1;
            g[i__ - 1] = t2 * 8. - x[i__ - 1] * 16. * t1;
            /* L22: */
        }
        g[n - 1] = t1 * 8.;
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    pub fn test_val(){
        let mut f = double::NAN;
        let x = vec![5.0, 17.0];
        let mut g = vec![double::NAN;x.len()];
        objective_function(&x, &mut f);
        gradient_function(&x, &mut g);
        println!("Objective value: {}", f);
        println!("Gradient function: {:?}", g);
    }
}