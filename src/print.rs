use crate::common::*;
pub fn errclb(
    n: &mut integer,
    m: &mut integer,
    factr: &mut double,
    l: &mut [double],
    u: &mut [double],
    nbd: &mut [integer],
    task: &mut integer,
    info: &mut integer,
    k: &mut integer,
    task_len: usize,
) -> integer {
    if *n <= 0 {
        *task = TaskStatus::ERROR_N0.status();
    }
    if *m <= 0 {
        *task = TaskStatus::ERROR_M0.status();
    }
    if *factr < 0.0 {
        *task = TaskStatus::ERROR_FACTR.status();
    }

    for i in 0..*n as usize {
        if nbd[i] < 0 || nbd[i] > 3 {
            *task = TaskStatus::ERROR_NBD.status();
            *info = -6;
            *k = i as integer;
        }
        if nbd[i] == 2 {
            if l[i] > u[i] {
                *task = TaskStatus::ERROR_FEAS.status();
                *info = -7;
                *k = i as integer;
            }
        }
    }

    return 0;
}

pub fn prn3lb(
    n: &mut integer,
    x: &mut double,
    f: &mut double,
    task: &mut integer,
    iprint: &mut integer,
    info: &mut integer,
    itfile: &mut fileType,
    iter: &mut integer,
    nfgv: &mut integer,
    nintol: &mut integer,
    nskip: &mut integer,
    nact: &mut integer,
    sbgnrm: &mut double,
    time: &mut double,
    nseg: &mut integer,
    word: &mut integer,
    iback: &mut integer,
    stp: &mut double,
    xstep: &mut double,
    k: &mut integer,
    cachyt: &mut double,
    sbtime: &mut double,
    lnscht: &mut double,
    task_len: &mut integer,
    word_len: &mut integer,
) {
    let mut i__1: integer;
    let mut i__: integer;

    if TaskStatus::is_error(*task) == 1 {
        //Error handling (L999)
    }
    if *iprint >= 0 {
        println!("           * * * ");
        println!("Tit   = total number of iterations");
        println!("Tnf   = total number of function evaluations");
        println!("Tnint = total number of segments explored during Cauchy searches");
        println!("Skip  = number of BFGS updates skipped");
        println!("Nact  = number of active bounds at final generalized Cauchy point");
        println!("Projg = norm of the final projected gradient");
        println!("F     = final function value");
        println!("           * * * ");

        println!("   N    Tit   Tnf  Tnint  Skip  Nact      Projg        F");
        println!(
            "{:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>12.2e} {:>12.5e}",
            *n, *iter, *nfgv, *nintol, *nskip, *nact, *sbgnrm, *f
        );
        todo!()
    }
}
