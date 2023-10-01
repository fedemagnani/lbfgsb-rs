//stuff of lbfgsb.h
pub type integer = i64;
pub type double = f64;
pub type fileType = String;
use std::time::{SystemTime, Duration};

pub fn timer(ttime: &mut f64) -> i32 {
    let start_time = SystemTime::now();
    *ttime = start_time.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs_f64();
    return 0;
}

pub (crate) const DBL_EPSILON:double= 2.2e-16;

pub enum Word{
    DEFAULT,
    CON,
    BND,
    TNT,
}
impl Word {
    pub fn word(&self)->integer{
        match self{
            Word::DEFAULT=>0,
            Word::CON=>1,
            Word::BND=>2,
            Word::TNT=>3,
        }
    }
}

pub enum TaskStatus{
    START,
    NEW_X,
    ABNORMAL,
    RESTART,
    FG,
    FG_END,
    FG_LN,
    FG_LNSRCH,
    FG_ST,
    FG_START,
    CONVERGENCE,
    CONVERGENCE_END,
    CONV_GRAD,
    CONV_F,
    STOP,
    STOP_END,
    STOP_CPU,
    STOP_ITER,
    STOP_GRAD,
    WARNING,
    WARNING_END,
    WARNING_ROUND,
    WARNING_XTOL,
    WARNING_STPMAX,
    WARNING_STPMIN,
    ERROR,
    ERROR_END,
    ERROR_SMALLSTP,
    ERROR_LARGESTP,
    ERROR_INITIAL,
    ERROR_FTOL,
    ERROR_GTOL,
    ERROR_XTOL,
    ERROR_STP0,
    ERROR_STP1,
    ERROR_N0,
    ERROR_M0,
    ERROR_FACTR,
    ERROR_NBD,
    ERROR_FEAS
}
impl TaskStatus{
    pub fn status(&self)->integer{
        match self{
            TaskStatus::START=>1,
            TaskStatus::NEW_X=>2,
            TaskStatus::ABNORMAL=>3,
            TaskStatus::RESTART=>4,
            TaskStatus::FG=>10,
            TaskStatus::FG_END=>15,
            TaskStatus::FG_LN=>11,
            TaskStatus::FG_LNSRCH=>11,
            TaskStatus::FG_ST=>12,
            TaskStatus::FG_START=>12,
            TaskStatus::CONVERGENCE=>20,
            TaskStatus::CONVERGENCE_END=>25,
            TaskStatus::CONV_GRAD=>21,
            TaskStatus::CONV_F=>22,
            TaskStatus::STOP=>30,
            TaskStatus::STOP_END=>40,
            TaskStatus::STOP_CPU=>31,
            TaskStatus::STOP_ITER=>32,
            TaskStatus::STOP_GRAD=>33,
            TaskStatus::WARNING=>100,
            TaskStatus::WARNING_END=>110,
            TaskStatus::WARNING_ROUND=>101,
            TaskStatus::WARNING_XTOL=>102,
            TaskStatus::WARNING_STPMAX=>103,
            TaskStatus::WARNING_STPMIN=>104,
            TaskStatus::ERROR=>200,
            TaskStatus::ERROR_END=>240,
            TaskStatus::ERROR_SMALLSTP=>201,
            TaskStatus::ERROR_LARGESTP=>202,
            TaskStatus::ERROR_INITIAL=>203,
            TaskStatus::ERROR_FTOL=>204,
            TaskStatus::ERROR_GTOL=>205,
            TaskStatus::ERROR_XTOL=>206,
            TaskStatus::ERROR_STP0=>207,
            TaskStatus::ERROR_STP1=>208,
            TaskStatus::ERROR_N0=>209,
            TaskStatus::ERROR_M0=>210,
            TaskStatus::ERROR_FACTR=>211,
            TaskStatus::ERROR_NBD=>212,
            TaskStatus::ERROR_FEAS=>213
        }
    }
    pub fn is_fg(x:integer)->integer{
        if x>=TaskStatus::FG.status() && x<=TaskStatus::FG_END.status(){
            1
        }else{
            0
        }
    }
    pub fn is_converged(x:integer)->integer{
        if x>=TaskStatus::CONVERGENCE.status() && x<=TaskStatus::CONVERGENCE_END.status(){
            1
        }else{
            0
        }
    }
    pub fn is_stop(x:integer)->integer{
        if x>=TaskStatus::STOP.status() && x<=TaskStatus::STOP_END.status(){
            1
        }else{
            0
        }
    }
    pub fn is_warning(x:integer)->integer{
        if x>=TaskStatus::WARNING.status() && x<=TaskStatus::WARNING_END.status(){
            1
        }else{
            0
        }
    }
    pub fn is_error(x:integer)->integer{
        if x>=TaskStatus::ERROR.status() && x<=TaskStatus::ERROR_END.status(){
            1
        }else{
            0
        }
    }
}