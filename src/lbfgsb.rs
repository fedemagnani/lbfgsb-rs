use core::num;
use ndarray::{prelude::*, OwnedRepr};
use std::{collections::HashMap, fmt::Display, ops::Mul};
// use ndarray_linalg::Inverse;

use crate::{common::*, print};
use log::{info, warn};
// The function returns the image and the gradient

pub trait AdditionalMethodsNdarray {
    fn tril(&self, k: i32) -> Self; //returns the elements on and below the kth diagonal of A. If k=-1, it returns the lower triangular part of A excluding the main diagonal. If k=0 it returns the lower triangular part of A including the main diagonal. If k=1 it returns the lower triangular part of A including the main diagonal and the diagonal above it, and so on.
    fn inv(&self) -> Self;
}

impl AdditionalMethodsNdarray for ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    fn tril(&self, k: i32) -> Self {
        let mut A = self.clone();
        let (n, m) = A.dim();
        for i in 0..n {
            for j in 0..m {
                match k.cmp(&0) {
                    std::cmp::Ordering::Less => {
                        if j + ((-k) as usize) > i {
                            A[[i, j]] = 0.0;
                        }
                    }
                    std::cmp::Ordering::Equal => {
                        if j > i {
                            A[[i, j]] = 0.0;
                        }
                    }
                    std::cmp::Ordering::Greater => {
                        if j > i + k as usize {
                            A[[i, j]] = 0.0;
                        }
                    }
                }
            }
        }
        A
    }

    fn inv(&self) -> Self {
        let mut A = self.clone();
        let (n, m) = A.dim();
        if n != m {
            panic!("The matrix is not square");
        }
        let mut I = Array2::<f64>::eye(n);
        for i in 0..n {
            let mut pivot = A[[i, i]];
            if pivot == 0.0 {
                let mut j = i + 1;
                while j < n && pivot == 0.0 {
                    pivot = A[[j, i]];
                    j += 1;
                }
                if pivot == 0.0 {
                    panic!("The matrix is singular");
                }
                for k in 0..n {
                    let tmp = A[[i, k]];
                    A[[i, k]] = A[[j - 1, k]];
                    A[[j - 1, k]] = tmp;
                    let tmp = I[[i, k]];
                    I[[i, k]] = I[[j - 1, k]];
                    I[[j - 1, k]] = tmp;
                }
            }
            for j in 0..n {
                A[[i, j]] = A[[i, j]] / pivot;
                I[[i, j]] = I[[i, j]] / pivot;
            }
            for j in 0..n {
                if j != i {
                    let mut ratio = A[[j, i]];
                    for k in 0..n {
                        A[[j, k]] = A[[j, k]] - ratio * A[[i, k]];
                        I[[j, k]] = I[[j, k]] - ratio * I[[i, k]];
                    }
                }
            }
        }
        I
    }
}

pub enum OptimizationError {
    InputError(String),
    GenericError(String),
}
impl Display for OptimizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationError::InputError(s) => write!(f, "{}", s),
            OptimizationError::GenericError(s) => write!(f, "{}", s),
        }
    }
}

pub struct OptimizerOptions {
    pub m: integer,         //the maximum number of stored L-BFGS iteration pairs.
    pub tol: double,        //the tolerance for convergence, typically 1e-5
    pub max_iters: integer, //the maximum number of iterations
    pub display: bool,      //whether to display the progress of the optimization
    pub xhist: bool,        //whether to return the history of the iterations
}
impl Default for OptimizerOptions {
    fn default() -> Self {
        OptimizerOptions {
            m: 5,
            tol: 1e-5,
            max_iters: 500,
            display: false,
            xhist: false,
        }
    }
}

pub struct LBFGSB;
impl LBFGSB {
    pub fn get_optimality(
        x: &Vec<double>,
        g: &Vec<double>,
        l: &Vec<double>,
        u: &Vec<double>,
    ) -> double {
        let mut projected_g = Array::from(x.clone()) - Array::from(g.clone());
        for i in 0..x.len() {
            if projected_g[i] < l[i] {
                projected_g[i] = l[i];
            } else if projected_g[i] > u[i] {
                projected_g[i] = u[i];
            }
        }
        let projected_g = projected_g - Array::from(x.clone());
        let opt = match projected_g
            .mapv(|x| x.abs())
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
        {
            Some(opt) => opt,
            None => panic!("Error in the computation of the optimality"),
        };
        opt
    }

    pub fn get_breakpoints(
        x: &Vec<double>,
        g: &Vec<double>,
        l: &Vec<double>,
        u: &Vec<double>,
    ) -> (Vec<double>, Vec<double>, Vec<integer>) {
        let n = x.len();
        let mut t = vec![0.0; n];
        let mut d = -Array::from(g.clone());
        for i in 0..n {
            if g[i] < 0.0 {
                t[i] = (x[i] - u[i]) / g[i];
            } else if g[i] > 0.0 {
                t[i] = (x[i] - l[i]) / g[i];
            } else {
                t[i] = double::MAX;
            }
            if t[i] < double::EPSILON {
                d[i] = 0.0
            }
        }
        // we sort the vector t in ascending order and we return the vector of the associated indices
        let mut F: Vec<integer> = (0..n as integer).collect();
        F.sort_by(|a, b| t[*a as usize].partial_cmp(&t[*b as usize]).unwrap());
        (t, d.to_vec(), F)
    }

    pub fn get_cauchy_point(
        x: &Vec<double>,
        g: &Vec<double>,
        l: &Vec<double>,
        u: &Vec<double>,
        theta: double,
        W: &ArrayBase<OwnedRepr<double>, Dim<[usize; 2]>>,
        M: &ArrayBase<OwnedRepr<double>, Dim<[usize; 2]>>,
    ) -> (Vec<double>, Vec<double>) {
        let (tt, d, F) = LBFGSB::get_breakpoints(x, g, l, u);
        let mut d = Array::from(d);
        let mut xc = x.clone();
        let mut p = W.t().dot(&d);
        let mut c = Array::from(vec![0.0; W.ncols()]);
        let mut fp = -&d.t().dot(&d);
        let mut fpp = -theta * fp - p.t().dot(M).dot(&p);
        let fpp0 = -theta * fp;
        let mut dt_min = -fp / fpp;
        let mut t_old = 0.0;
        let mut i = 0;
        for j in 0..x.len() {
            i = j;
            if F[i] > 0 {
                break;
            }
        }
        let mut b = F[i];
        let mut t = tt[b as usize];
        let mut dt = t - t_old;
        // here i is expected to start from 0, while in matlab conde it starts from 1
        while dt_min > dt && i <= x.len() {
            if d[b as usize] > 0.0 {
                xc[b as usize] = u[b as usize];
            } else if d[b as usize] < 0.0 {
                xc[b as usize] = l[b as usize];
            }
            let zb = xc[b as usize] - x[b as usize];
            c = c + dt * &p;
            let gb = g[b as usize];
            // We take the bth row of W
            let wbt = W.slice(s![b as usize, ..]);
            fp = fp + dt * fpp + gb * gb + theta * gb * zb - gb * wbt.dot(M).dot(&c);
            fpp = fpp
                - theta * gb * gb
                - 2.0 * gb * wbt.dot(M).dot(&p)
                - gb * gb * wbt.dot(M).dot(&wbt.t());
            fpp = fpp.max(double::EPSILON * fpp0);
            p = p + gb * &wbt.t();
            d[b as usize] = 0.0;
            dt_min = -fp / fpp;
            t_old = t;
            i = i + 1;
            if i <= x.len() {
                b = F[i];
                t = tt[b as usize];
                dt = t - t_old;
            }
        }
        dt_min = dt_min.max(0.0);
        t_old = t_old + dt_min;
        for j in i..xc.len() {
            let idx = F[j] as usize;
            xc[idx] = x[idx] + t_old * d[idx];
        }
        c = c + dt_min * &p;
        (xc, c.to_vec())
    }

    pub fn check_input(
        x0: &Vec<double>,
        l: &Vec<double>,
        u: &Vec<double>,
    ) -> Result<Vec<double>, OptimizationError> {
        let n = x0.len();
        if l.len() != n || u.len() != n {
            return Err(OptimizationError::InputError(
                format!("The length of the input vector is {}, but the length of the lower bound is {} and the length of the upper bound is {}",n,l.len(),u.len())
            ));
        }
        for i in 0..n {
            if l[i] > u[i] {
                return Err(OptimizationError::InputError(format!(
                    "The lower bound at index {} is greater than the upper bound ({} > {})",
                    i, l[i], u[i]
                )));
            }
        }
        // We constrain the initial x0 to be in bounds
        let mut x0 = x0.clone();
        let mut in_bounds = true;
        for i in 0..n {
            if x0[i] < l[i] {
                in_bounds = false;
                x0[i] = l[i];
            } else if x0[i] > u[i] {
                in_bounds = false;
                x0[i] = u[i];
            }
        }
        if !in_bounds {
            info!("The initial point was not in bounds, it has been projected into the bounds");
        }
        Ok(x0)
    }

    pub fn subspace_min(
        x: &Vec<double>,
        g: &Vec<double>,
        l: &Vec<double>,
        u: &Vec<double>,
        xc: &Vec<double>,
        c: &Vec<double>,
        theta: double,
        W: &ArrayBase<OwnedRepr<double>, Dim<[usize; 2]>>,
        M: &ArrayBase<OwnedRepr<double>, Dim<[usize; 2]>>,
    ) -> (Vec<double>, bool) {
        let mut line_search_flag = true;

        let n = x.len();
        let mut free_vars_idx: Vec<usize> = Vec::new();
        // HOW TO CREATE EMPTY MATRIX AND APPEND COLUMNS WITH FIXED ROW NUMBER
        let mut z = vec![];
        // for (i, &xc_i) in xc.iter().enumerate() {
        //     if xc_i != u[i] && xc_i != l[i] {
        //         free_vars_idx.push(i);
        //         // let mut unit = Array::zeros(n);
        //         let mut unit = vec![0.0; n];
        //         unit[i] = 1.0;
        //         z.push(unit);
        //     }
        // }
        for i in 0..n{
            if xc[i] != u[i] && xc[i] != l[i] {
                free_vars_idx.push(i);
                // let mut unit = Array::zeros(n);
                let mut unit = vec![0.0; n];
                unit[i] = 1.0;
                z.push(unit);
            }
        }
        let num_free_vars = free_vars_idx.len();
        if num_free_vars == 0 {
            let xbar = xc.clone();
            line_search_flag = false;
            return (xbar, line_search_flag);
        }
        let z = match Array2::from_shape_vec((n, free_vars_idx.len()), z.concat()) {
            Ok(z) => z,
            Err(e) => panic!("Error in the creation of the matrix z: {}", e),
        };
        let WTZ: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = W.t().dot(&z);
        let c = Array::from(c.clone());
        let rr = Array::from(g.clone())
            + theta * (Array::from(xc.clone()) - Array::from(x.clone()))
            - W.dot(&M.dot(&c));
        let mut r: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> =
            Array::from(vec![0.0; num_free_vars]);
        // for (i, &idx) in free_vars_idx.iter().enumerate() {
        //     r[i] = rr[idx];
        // }
        for i in 0..num_free_vars{
            r[i] = rr[free_vars_idx[i]];
        }
        let invtheta = 1.0 / theta;
        let v: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = M.dot(&(WTZ.dot(&r)));
        let N: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = invtheta * WTZ.dot(&WTZ.t());
        // println!("dim N = {:?}", N.dim());
        let mut eye: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array2::zeros(N.dim());
        for a_ii in eye.diag_mut() {
            *a_ii = 1.0;
        }
        // println!("dim M = {:?}", M.dim());
        let N: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = eye - M.dot(&N);
        // let v: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = N / v;
        let v: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = N.inv().dot(&v);
        let du = -invtheta * r - invtheta.powi(2) * WTZ.t().dot(&v);
        let du = du.into_raw_vec();
        let du = Array::from(du);

        let alpha_star = LBFGSB::find_alpha(l, u, xc, &du.to_vec(), &free_vars_idx);

        let d_star = alpha_star * du;
        let mut xbar = xc.clone();
        for (i, &idx) in free_vars_idx.iter().enumerate() {
            xbar[idx] = xbar[idx] + d_star[i];
        }
        (xbar, line_search_flag)
    }

    pub fn find_alpha(
        l: &Vec<double>,
        u: &Vec<double>,
        xc: &Vec<double>,
        du: &Vec<double>,
        free_vars_idx: &Vec<usize>,
    ) -> double {
        let mut alpha_star: double = 1.0;
        let n = free_vars_idx.len();
        for i in 0..n {
            let idx = free_vars_idx[i];
            if du[i] > 0.0 {
                alpha_star = alpha_star.min((u[idx] - xc[idx]) / du[i]);
            } else {
                alpha_star = alpha_star.min((l[idx] - xc[idx]) / du[i]);
            }
        }
        alpha_star
    }

    pub fn strong_wolfe<F>(
        func: F,
        x0: &Vec<double>,
        f0: double,
        g0: &Vec<double>,
        p: &Vec<double>,
    ) -> double
    where
        F: Fn(Vec<double>) -> (double, Vec<double>),
    {
        let c1 = 1e-4;
        let c2 = 0.9;
        let alpha_max = 2.5;
        let mut alpha_im1 = 0.0;
        let mut alpha_i = 1.0;
        let mut f_im1 = f0;
        let g0 = Array::from(g0.clone());
        let p = Array::from(p.clone());
        let x0 = Array::from(x0.clone());
        let dphi0 = g0.dot(&p);
        let mut i = 0;
        let max_iter = 20;
        let mut alpha = 0.0;
        loop {
            let x = &x0 + alpha_i * &p;
            let (f_i, g_i) = func(x.to_vec().clone());
            if f_i > f0 + c1 * alpha_i * dphi0 || (i > 0 && f_i >= f_im1) {
                return LBFGSB::alpha_zoom(
                    func,
                    &x0.to_vec(),
                    f0,
                    &g0.to_vec(),
                    &p.to_vec(),
                    alpha_im1,
                    alpha_i,
                );
            }
            let dphi = Array::from(g_i).dot(&p);
            if dphi.abs() <= -c2 * dphi0 {
                alpha = alpha_i;
                return alpha;
            }
            if dphi >= 0.0 {
                return LBFGSB::alpha_zoom(
                    func,
                    &x0.to_vec(),
                    f0,
                    &g0.to_vec(),
                    &p.to_vec(),
                    alpha_i,
                    alpha_im1,
                );
            }
            alpha_im1 = alpha_i;
            f_im1 = f_i;
            alpha_i = alpha_i + 0.8 * (alpha_max - alpha_i);

            if i > max_iter {
                alpha = alpha_i;
                return alpha;
            }
            i += 1;
        }
    }

    pub fn alpha_zoom<F>(
        func: F,
        x0: &Vec<double>,
        f0: double,
        g0: &Vec<double>,
        p: &Vec<double>,
        alpha_lo: double,
        alpha_hi: double,
    ) -> double
    where
        F: Fn(Vec<double>) -> (double, Vec<double>),
    {
        let c1 = 1e-4;
        let c2 = 0.9;
        let mut i = 0;
        let max_iters = 20;
        let g0 = Array::from(g0.clone());
        let p = Array::from(p.clone());
        let x0 = Array::from(x0.clone());
        let dphi0 = g0.t().dot(&p);
        let mut alpha_hi = alpha_hi;
        let mut alpha_lo = alpha_lo;
        let mut alpha = 0.0;
        loop {
            let alpha_i = (alpha_lo + alpha_hi) / 2.0;
            alpha = alpha_i;
            let x = &x0 + alpha_i * &p;
            let (f_i, g_i) = func(x.to_vec().clone());
            let x_lo = &x0 + alpha_lo * &p;
            let (f_lo, _) = func(x_lo.to_vec().clone());
            if f_i > f0 + c1 * alpha_i * dphi0 || f_i >= f_lo {
                alpha_hi = alpha_i;
            } else {
                let dphi = Array::from(g_i).dot(&p);
                if dphi.abs() <= -c2 * dphi0 {
                    alpha = alpha_i;
                    return alpha;
                }
                if dphi * (alpha_hi - alpha_lo) >= 0.0 {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_i;
            }
            i += 1;
            if i > max_iters {
                alpha = alpha_i;
                return alpha;
            }
        }
    }

    pub fn LBFGSB<F>(
        func: F,
        x0: &Vec<double>,
        l: &Vec<double>,
        u: &Vec<double>,
        options: Option<OptimizerOptions>,
    ) -> (Vec<double>, Option<Vec<Vec<double>>>)
    where
        F: Fn(Vec<double>) -> (double, Vec<double>),
    {
        let x0 = LBFGSB::check_input(x0, l, u);
        let x0 = match x0 {
            Ok(x0) => x0,
            Err(e) => panic!("Error in the input: {}", e),
        };
        let options = match options {
            Some(options) => options,
            None => OptimizerOptions::default(),
        };
        let n = x0.len();
        let mut Y = Array2::<double>::zeros((n, 0));
        let mut S = Array2::<double>::zeros((n, 0));
        // let mut  Y = vec![];
        // let mut S = vec![];
        let mut W = Array2::<double>::zeros((n, 2 * options.m as usize));
        let mut M = Array2::<double>::zeros((2 * options.m as usize, 2 * options.m as usize));
        let mut theta = 1.0;
        let mut x = x0;
        let (f, mut g) = func(x.clone());
        let mut k = 0;
        let opt = LBFGSB::get_optimality(&x, &g, l, u);
        if options.display {
            println!(
                "Iteration {}:\n\tf = {}\n\tg = {:?}\n\toptimality = {:?}",
                k, f, g, opt
            );
        }
        let mut xhist = vec![];
        if options.xhist {
            xhist.push(x.clone());
        }
        while LBFGSB::get_optimality(&x, &g, l, u) > options.tol && k < options.max_iters {
            let x_old = x.clone();
            let g_old = &g;
            let (xc, c) = LBFGSB::get_cauchy_point(&x, &g, l, u, theta, &W, &M);
            let (xbar, line_search_flag) =
                LBFGSB::subspace_min(&x, &g, l, u, &xc, &c, theta, &W, &M);
            let mut alpha = 1.0;
            if line_search_flag {
                alpha = LBFGSB::strong_wolfe(
                    &func,
                    &x,
                    f,
                    &g,
                    &(&Array::from(xbar.clone()) - &Array::from(x.clone())).to_vec(),
                );
            }
            x = (Array::from(x.clone()) + alpha * (Array::from(xbar) - Array::from(x.clone())))
                .to_vec();

            let (f, g) = func(x.clone());
            let y = &Array::from(g.clone()) - &Array::from(g_old.clone());
            let s = &Array::from(x.clone()) - &Array::from(x_old.clone());
            let curv = s.dot(&y).abs();
            if curv < double::EPSILON {
                warn!(" warning: negative curvature detected\n");
                warn!("          skipping L-BFGS update\n");
                k = k + 1;
                continue;
            }
            let reshaped_y = match y.clone().into_shape((n, 1)) {
                Ok(y) => y,
                Err(e) => panic!("Error in the reshaping of y: {}", e),
            };
            let reshaped_s = match s.clone().into_shape((n, 1)) {
                Ok(s) => s,
                Err(e) => panic!("Error in the reshaping of s: {}", e),
            };
            if k < options.m {
                Y = match ndarray::concatenate(Axis(1), &[Y.view(), reshaped_y.view()]) {
                    Ok(Y) => Y,
                    Err(e) => panic!("Error in the concatenation of Y: {}", e),
                };
                S = match ndarray::concatenate(Axis(1), &[S.view(), reshaped_s.view()]) {
                    Ok(S) => S,
                    Err(e) => panic!("Error in the concatenation of S: {}", e),
                };
            } else {
                let yyyyy = Y.clone();
                let assignable_y = yyyyy.slice(s![.., 1..]);
                Y.slice_mut(s![.., 0..options.m as usize - 1])
                    .assign(&assignable_y);
                let sssss = S.clone();
                let assignable_s = sssss.slice(s![.., 1..]);
                S.slice_mut(s![.., 0..options.m as usize - 1])
                    .assign(&assignable_s);
                Y.slice_mut(s![.., options.m as usize - 1]).assign(&y);
                S.slice_mut(s![.., options.m as usize - 1]).assign(&s);
            }
            let a_y = Array::from(y.clone());
            let a_s = Array::from(s.clone());
            theta = a_y.dot(&a_y) / a_y.dot(&a_s);
            W = match ndarray::concatenate(Axis(1), &[Y.view(), (theta * S.clone()).view()]) {
                Ok(W) => W,
                Err(e) => panic!("Error in the concatenation of W: {}", e),
            };
            let A = S.t().dot(&Y);
            let L = A.tril(-1);
            let D = -1.0 * Array::from_diag(&A.diag());
            let M1 = ndarray::concatenate![Axis(1), D, L.t()];
            let M2 = ndarray::concatenate![Axis(1), L, theta * S.t().dot(&S)];
            
            let MM = ndarray::concatenate![Axis(0), M1, M2];
            M = MM.inv();
            // println!("OK");
            k += 1;
            if options.xhist {
                xhist.push(x.clone());
            }
            if options.display {
                println!(
                    "Iteration {}:\n\tf = {}\n\tg = {:?}\n\toptimality = {:?}",
                    k,
                    f,
                    g,
                    LBFGSB::get_optimality(&x, &g, l, u)
                );
            }
        }
        if k == options.max_iters {
            warn!("Maximum number of iterations reached");
        }
        if LBFGSB::get_optimality(&x, &g, l, u) < options.tol {
            info!("Optimization terminated successfully");
        }
        (x, Some(xhist))
    }
    // Returns the input vector and the history of the iterations, according to options
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;

    fn apply<F>(f: F, x: i32) -> i32
    where
        F: Fn(i32) -> i32,
    {
        f(x)
    }
    fn foo(x: i32) -> i32 {
        x * 2
    }
    #[test]
    fn test_apply() {
        let result = apply(foo, 3);
        assert_eq!(result, 6);
    }

    use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
    #[test]
    fn display_matrix() {
        // We expect w to be a n times 2m matrix
        let n = 5;
        let m = 5;
        let w: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array2::zeros((2 * n, 2 * m));
        println!("w = {:#?}", w);
        let second_row_of_w: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 1]>> =
            w.slice(s![1, ..]);
        println!("second_row_of_w = {:#?}", second_row_of_w.t());
        // We transform it as a column vector
        let quad = second_row_of_w.t().dot(&w).dot(&second_row_of_w);
        println!("quad = {:#?}", quad);
        // let a = second_row_of_w.dot(&w);
        // println!("a = {:#?}", a);
    }

    #[test]
    fn build_empty_matrix_to_be_accumulated() {
        let n = 5;
        let mut Y = Array2::<double>::zeros((n, 0));
        let mut S = Array2::<double>::zeros((n, 0));
        println!("Y = {:#?}", Y);
        // we append a column vector to y
        let unit = Array::from(vec![1.0; n]);
        let zeros = Array::from(vec![0.0; n]);
        let Y = ndarray::concatenate(
            Axis(1),
            &[Y.view(), unit.into_shape((n, 1)).unwrap().view()],
        )
        .unwrap();
        println!("Y = {:#?}", Y);
    }

    #[test]
    fn sort_table() {
        let mut data: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![2.0, 8.0, 9.0],
        ];

        println!("{:?}", data);
    }

    #[test]
    fn concactenate_matrix() {
        let Z = Array2::<double>::zeros((5, 0));
        let ones = Array::from(vec![1.0; 5]);
        let ones_higher_dim = ones.clone().into_shape((ones.len(), 1)).unwrap();
        let z_2 = ndarray::concatenate(Axis(1), &[Z.view(), ones_higher_dim.view()]);
        println!("z_2 = {:#?}", z_2);
    }

    #[test]
    fn build_matrix() {
        let mut z = vec![];
        let col_num = 5;
        for i in 0..col_num {
            let mut unit = vec![0.0; col_num];
            unit[i] = 1.0;
            z.push(unit);
        }
        println!("z = {:#?}", z);
        // We create a matrix where each column is a vector in z
        let z = Array2::from_shape_vec((col_num, col_num), z.concat()).unwrap();
        println!("z = {:#?}", z);
    }

    #[test]
    fn build_matrix_2() {
        let mut z = vec![];
        let col_num = 5;
        for i in 0..col_num {
            let mut unit = vec![0.0; col_num];
            unit[i] = 1.0;
            z.push(unit);
        }
    }

    #[test]
    fn non_square_eye() {
        let n = 5;
        let m = 3;
        let mut eye = Array2::zeros((n, m));
        for a_ii in eye.diag_mut() {
            *a_ii = 1.0;
        }
        println!("eye = {:#?}", eye);
    }

    #[test]
    fn sorted_indices() {
        let t = vec![3.2737, 1.5738, 9.928374, 8.4];
        let n = t.len();
        let mut index: Vec<usize> = (0..n).collect();
        index.sort_by(|a, b| t[*a].partial_cmp(&t[*b]).unwrap());
        println!("{:?}", index);
    }

    #[test]
    pub fn create_lower_triang_matrix() {
        use ndarray::{array, Array2};
        let A = Array2::<f64>::ones((3, 3));

        let L_1 = A.tril(-1);
        let L_2 = A.tril(0);
        let L_3 = A.tril(1);

        assert_eq!(
            L_1,
            array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        );
        assert_eq!(
            L_2,
            array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        );
        assert_eq!(
            L_3,
            array![[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
    }

    #[test]
    pub fn create_diagonal_matrix(){
        let d = vec![1.0;5];
        let D = Array::from_diag(&Array::from(d));
        println!("D = {:#?}",D);
    }

    use env_logger;
    #[test]
    pub fn test_optimization(){
        // We set the log level to debug
        let _ = env_logger::builder().filter_level(log::LevelFilter::Debug).try_init();
        
        pub fn dummy_func(x: Vec<double>) -> (double, Vec<double>) {
            let f = (x[0]-1.0).powi(2) + x[1].powi(2);
            let g = vec![2.0 *( x[0]-1.0), 2.0 * x[1]];
            (f, g)
        }
        let x0 = vec![2.0, 3.0];
        let l = vec![-5.0, -5.0];
        let u = vec![5.0, 5.0];
        let options = OptimizerOptions::default();
        let (x, _) = LBFGSB::LBFGSB(dummy_func, &x0, &l, &u, Some(options));
        println!("x = {:#?}",x);
    }

    #[test]
    pub fn divide_matrix(){
        let matrix = Array2::<double>::ones((5,5));
        let vec = Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let res = matrix / vec;
        println!("res = {:#?}",res);
    }

    #[test]
    pub fn compute_inv_matrix(){
        let matrix: ArrayBase<OwnedRepr<double>, Dim<[usize; 2]>> = array![
            [1.0,2.0,6.0,56.0,6.0,66.0],
            [77.0,7.0,5.0,6.0,66.0,23.0],
            [4.0,556.0,778.0,999.0,876.0,56.0],
            [666.0,54.0,5.0,44.0,44.0,334.0],
            [435.0,54.0,5.0,5.0,332.0,456.0],
            [215.0,654.0,67.0,8.0,77.0,5.0]
        ];

        let inv_matrix = matrix.inv();
        println!("inv_matrix = {:#?}",inv_matrix);
    }
}

// use crate::common::*;
// use crate::print::*;

// pub struct LBFGSB {
//     i__: integer,
//     k: integer,
//     gd: double,
//     dr: double,
//     rr: double,
//     dtd: double,
//     col: integer,
//     tol: double,
//     wrk: integer,
//     stp: double,
//     cpu1: double,
//     cpu2: double,
//     head: integer,
//     fold: double,
//     nact: integer,
//     ddum: double,
//     info: integer,
//     nseg: integer,
//     time: double,
//     nfgv: integer,
//     ifun: integer,
//     iter: integer,
//     word: integer,
//     time1: double,
//     time2: double,
//     iback: integer,
//     gdold: double,
//     nfree: integer,
//     boxed: integer,
//     itail: integer,
//     theta: double,
//     dnorm: double,
//     nskip: integer,
//     iword: integer,
//     xstep: double,
//     stpmx: double,
//     ileave: integer,
//     cachyt: double,
//     itfile: integer,
//     epsmch: double,
//     updatd: integer,
//     sbtime: double,
//     prjctd: integer,
//     iupdat: integer,
//     sbgnrm: double,
//     cnstnd: integer,
//     nenter: integer,
//     lnscht: double,
//     nintol: integer,
// }
// // We do not return anything, we just edit the mutable references of the input point and of the image
// // Variables declared static survive to the scope of the function. A hack could be defining those values as keys of the structure LBFGSB
// impl LBFGSB {
//     pub fn setulb() {}
//     pub fn mainlb(
//         &mut self,
//         n: &mut integer,
//         m: &mut integer,
//         x: &mut Vec<double>,
//         l: &mut Vec<double>,
//         u: &mut Vec<double>,
//         nbd: &mut Vec<integer>,
//         f: &mut double,
//         g: &mut Vec<double>,
//         factr: &mut double, //tolerance in stopping criteria
//         pgtol: &mut double,
//         ws: &mut Vec<Vec<double>>,
//         wy: &mut Vec<Vec<double>>,
//         sy: &mut Vec<Vec<double>>,
//         ss: &mut Vec<Vec<double>>,
//         wt: &mut Vec<Vec<double>>,
//         wn: &mut Vec<Vec<double>>,
//         snd: &mut Vec<Vec<double>>,
//         z__: &mut Vec<double>,
//         r__: &mut Vec<double>,
//         d__: &mut Vec<double>,
//         t: &mut Vec<double>,
//         xp: &mut Vec<double>,
//         wa: &mut Vec<double>,
//         index: &mut Vec<integer>,
//         iwhere: &mut Vec<integer>,
//         indx2: &mut Vec<integer>,
//         task: &mut integer,
//         iprint: &mut integer,
//         csave: &mut integer,
//         lsave: &mut Vec<integer>,
//         isave: &mut Vec<integer>,
//         dsave: &mut Vec<double>,
//     ) {
//         let mut ws_dim1: integer = 0;
//         let mut ws_offset: integer = 0;
//         let mut wy_dim1: integer = 0;
//         let mut wy_offset: integer = 0;
//         let mut sy_dim1: integer = 0;
//         let mut sy_offset: integer = 0;
//         let mut ss_dim1: integer = 0;
//         let mut ss_offset: integer = 0;
//         let mut wt_dim1: integer = 0;
//         let mut wt_offset: integer = 0;
//         let mut wn_dim1: integer = 0;
//         let mut wn_offset: integer = 0;
//         let mut snd_dim1: integer = 0;
//         let mut snd_offset: integer = 0;
//         let mut i__1: integer = 0;
//         let mut d__1: double = 0.0;
//         let mut d__2: double = 0.0;
//         let mut o__1: Option<fileType> = None;

//         if *task == TaskStatus::START.status() {
//             let epsmch = DBL_EPSILON;
//             timer(&mut self.time1);
//             // Initialize counters and scalars when task='START'.
//             // For the limited memory BFGS matrices:
//             let mut col: integer = 0;
//             let mut head: integer = 1;
//             let mut theta: double = 1.0;
//             let mut iupdat: integer = 0;
//             let mut updatd: integer = 0;
//             let mut iback: integer = 0;
//             let mut itail: integer = 0;
//             let mut iword: integer = 0;
//             let mut nact: integer = 0;
//             let mut ileave: integer = 0;
//             let mut nenter: integer = 0;
//             let mut fold: double = 0.0;
//             let mut dnorm: double = 0.0;
//             let mut cpu1: double = 0.0;
//             let mut gd: double = 0.0;
//             let mut stpmx: double = 0.0;
//             let mut sbgnrm: double = 0.0;
//             let mut stp: double = 0.0;
//             let mut gdold: double = 0.0;
//             let mut dtd: double = 0.0;

//             // For operation counts:
//             let mut iter: integer = 0;
//             let mut nfgv: integer = 0;
//             let mut nseg: integer = 0;
//             let mut nintol: integer = 0;
//             let mut nskip: integer = 0;
//             let mut nfree: integer = *n;
//             let mut ifun: integer = 0;

//             // For stopping tolerance:
//             let mut tol: double = *factr * epsmch;

//             // For measuring running time:
//             let mut cachyt: double = 0.0;
//             let mut sbtime: double = 0.0;
//             let mut lnscht: double = 0.0;

//             // 'word' records the status of subspace solutions.
//             self.word = Word::DEFAULT.word();

//             // 'info' records the termination information.
//             let mut info: integer = 0;
//             let mut itfile: integer = 8;

//             // Note: no longer trying to write to file
//             // Check the input arguments for errors.
//             errclb(n, m, factr, &mut [l[1]], &mut [u[1]], &mut [nbd[1]], task, &mut info, &mut self.k, 60); //this will change the task status to error if there is any error
//             if TaskStatus::is_error(*task) == 1{

//             }
//         }
//     }
// }

// // impl LBFGSB{
// //     // setulb is the fucntion that is invoked for running LBFGSB (check driver1.c)
// //     pub fn setulb(n:integer, m:integer, x:Vec<double>, l:Vec<double>, u:Vec<double>, nbd:Vec<integer>, f:double, g:Vec<double>, factr:double, pgtol:double, wa:Vec<double>, iwa:Vec<integer>, task:integer, iprint:integer, csave:integer, lsave:Vec<integer>, isave:&mut Vec<integer>, dsave:Vec<double>)->integer{
// //         let mut i__1: integer;
// //         let mut ld: integer;
// //         let mut lr: integer;
// //         let mut lt: integer;
// //         let mut lz: integer;
// //         let mut lwa: integer;
// //         let mut lwn: integer;
// //         let mut lss: integer;
// //         let mut lxp: integer;
// //         let mut lws: integer;
// //         let mut lwt: integer;
// //         let mut lsy: integer;
// //         let mut lwy: integer;
// //         let mut lsnd: integer;
// //         if task==TaskStatus::START.status(){
// //             isave[1] = m*n;
// //             i__1 = m;
// //             isave[2] = i__1 * i__1;
// //             i__1 = m;
// //             isave[3] = i__1 * i__1 << 2;
// //             isave[4] = 1;
// //             /* ws      m*n */
// //             isave[5] = isave[4] + isave[1];
// //             /* wy      m*n */
// //             isave[6] = isave[5] + isave[1];
// //             /* wsy     m**2 */
// //             isave[7] = isave[6] + isave[2];
// //             /* wss     m**2 */
// //             isave[8] = isave[7] + isave[2];
// //             /* wt      m**2 */
// //             isave[9] = isave[8] + isave[2];
// //             /* wn      4*m**2 */
// //             isave[10] = isave[9] + isave[3];
// //             /* wsnd    4*m**2 */
// //             isave[11] = isave[10] + isave[3];
// //             /* wz      n */
// //             isave[12] = isave[11] + n;
// //             /* wr      n */
// //             isave[13] = isave[12] + n;
// //             /* wd      n */
// //             isave[14] = isave[13] + n;
// //             /* wt      n */
// //             isave[15] = isave[14] + n;
// //             /* wxp     n */
// //             isave[16] = isave[15] + n;
// //         }
// //         lws = isave[4];
// //         lwy = isave[5];
// //         lsy = isave[6];
// //         lss = isave[7];
// //         lwt = isave[8];
// //         lwn = isave[9];
// //         lsnd = isave[10];
// //         lz = isave[11];
// //         lr = isave[12];
// //         ld = isave[13];
// //         lt = isave[14];
// //         lxp = isave[15];
// //         lwa = isave[16];
// //         // LBFGSB::mainlb(n, m, vec![x[1]], vec![l[1]], vec![u[1]], vec![nbd[1]], f, vec![g[1]], factr, pgtol, vec![wa[lws as usize]], vec![wa[lwy as usize]], vec![wa[lsy as usize]], vec![wa[lss as usize]], vec![wa[lwt as usize]], vec![wa[lwn as usize]], vec![wa[lsnd as usize]],
// //         // wa[lz as usize], vec![wa[lr as usize]], vec![wa[ld as usize]], wa[lt as usize], wa[lxp as usize], wa[lwa as usize], vec![iwa[1]],
// //         // iwa[(n + 1) as usize], iwa[((n << 1) + 1) as usize], task, iprint, csave, vec![lsave[1]],
// //         // &mut vec![isave[22]], vec![dsave[1]]); /* (ftnlen)60, (ftnlen)60); */
// //     return 0;
// //     }
// //     // pub fn mainlb(n:integer, m:integer, x:Vec<double>, l:Vec<double>, u:Vec<double>, nbd:Vec<integer>, f:double, g:Vec<double>, factr:double, pgtol:double, ws:Vec<double>, wy:Vec<double>, sy:Vec<double>,ss:Vec<double>, wt:Vec<double>, wn:Vec<double>, snd:Vec<double>, z__:Vec<double>, r__:Vec<double>, d__:Vec<double>, t:Vec<double>, xp:Vec<double>, wa:Vec<double>, index:Vec<integer>, iwhere:Vec<integer>, indx2:Vec<integer>, task:integer, iprint:integer, csave:integer, lsave:Vec<integer>, isave:&mut Vec<integer>, dsave:Vec<double>){}

// // }
