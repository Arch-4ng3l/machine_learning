use std::fmt::Display;
use rand::random;


#[derive(Clone)]
struct Matrix {
    arr: Vec<Vec<f32>>,
    shape: [usize; 2],
}

impl Matrix {
    fn new(shape: &[usize]) -> Self {
        return Matrix{
            arr: vec![vec![0.0; shape[1]]; shape[0]],
            shape: [shape[0], shape[1]]
        };
    }
    fn random(shape: &[usize]) -> Self {
        let mut vec = Vec::new();
        for _ in 0..shape[0] {
            let mut v: Vec<f32> = Vec::new();
            for _ in 0..shape[1] {
                v.push(random());
            }
            vec.push(v);
        }
        return Matrix{
            arr: vec,
            shape: [shape[0], shape[1]]
        };
    }

    fn transpose(&self) -> Self {
        let mut matrix = Matrix::new(&[self.arr[0].len(), self.arr.len()]);
        for i in 0..self.arr.len() {
            for j in 0..self.arr[0].len() {
                matrix.arr[j][i] = self.arr[i][j];
            }
        }
        return matrix;
    }
    fn from(arr: Vec<Vec<f32>>) -> Self{
        return Matrix {
            shape: [arr.len(), arr[0].len()],
            arr,
        }
    }
    fn apply(&self, f: &dyn Fn(f32) -> f32) -> Self {
        let mut res = Matrix::new(&self.shape);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                res.arr[i][j] = f(self.arr[i][j]);
            }
        }
        return res
    }

    fn mean(self, len: f32) -> f32 {
        let sum = self.sum();
        return sum / len;
    }
    fn sum(self) -> f32{
        let mut sum = 0.0;
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                sum = sum + self.arr[i][j];
            }
        }
        return sum;
    }


    fn mul_const(&self, val: f32) -> Self {
        let mut res = Matrix::new(&self.shape);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                res.arr[i][j] = self.arr[i][j] * val;
            }
        }

        return res
    }
    fn matmul(&self, rhs: &Self) -> Self{
        assert!(self.shape[1] == rhs.shape[0]);
        let mut res = Matrix::new(&[self.shape[0], rhs.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                for k in 0..rhs.shape[0] {
                    res.arr[i][j] = res.arr[i][j] + rhs.arr[k][j] * self.arr[i][k];
                }
            }
        }
        return res
    }


    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1]);
        let mut res = Matrix::new(&self.shape);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                res.arr[i][j] = self.arr[i][j] * rhs.arr[i][j];
            }
        }

        return res

    }


    fn add(&self, rhs: &Self) -> Self {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1]);
        let mut res = Matrix::new(&self.shape);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                res.arr[i][j] = self.arr[i][j] + rhs.arr[i][j];
            }
        }
        return res

    }

    fn sub(&self, rhs: &Self) -> Self {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1]);
        let mut res = Matrix::new(&self.shape);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                res.arr[i][j] = self.arr[i][j] - rhs.arr[i][j];
            }
        }
        return res

    }

}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.arr.len() {
            s += "[ ";
            for j in 0..self.arr[0].len() {
                let val = self.arr[i][j];
                s += &format!("{val} ");
            }
            s += "]\n";
        }
        s += &format!("[{}, {}]", self.shape[0], self.shape[1]);
        write!(f, "{s}")
    }
}

trait Function {
    fn func(&self) -> &dyn Fn(f32) -> f32;
    fn derivative(&self) -> &dyn Fn(f32) -> f32;
}

fn relu(num: f32) -> f32 {
    if num > 0.0 {
        return num
    }

    return 0.0;
}
fn relu_deriv(num: f32) -> f32 {
    if num > 0.0 {
        return 0.0;
    }
    return 0.0
}

fn square(num: f32) -> f32 {
    return num * num;
}

fn mse(num: &Matrix, target: &Matrix) -> f32{
    let l = (num.shape[0] * num.shape[1]) as f32;
    return num.sub(&target).apply(&square).mean(l);
}

fn mse_deriv(num: &Matrix, target: &Matrix) -> Matrix{
    return num
        .sub(&target)
}

fn sigmoid(num: f32) -> f32 {
    return 1.0/ (1.0 + f32::exp(-num))
}
fn sigmoid_deriv(num: f32) -> f32 {
    return sigmoid(num) * (1.0 - sigmoid(num))
}

struct Sigmoid {}

impl Function for Sigmoid {
    fn func(&self) -> &dyn Fn(f32) -> f32 {
        return &sigmoid
    }
    fn derivative(&self) -> &dyn Fn(f32) -> f32 {
        return &sigmoid_deriv
    }
}

struct Linear {}

impl Function for Linear {
    fn func(&self) -> &dyn Fn(f32) -> f32 {
        return &|num: f32| -> f32 {
            return num;
        };
    }
    fn derivative(&self) -> &dyn Fn(f32) -> f32 {
        return &|_num: f32| -> f32 {
            return 1.0;
        };
    }
}

struct ReLU {}

impl Function for ReLU {
    fn func(&self) -> &dyn Fn(f32) -> f32 {
        return &relu;
    }
    fn derivative(&self) -> &dyn Fn(f32) -> f32 {
        return &relu_deriv;
    }
}

trait LossFunction {
    fn func(&self) -> &dyn Fn(&Matrix, &Matrix) -> f32;
    fn derivative(&self) -> &dyn Fn(&Matrix, &Matrix) -> Matrix;
}




struct MSE {}

impl LossFunction for MSE {
    fn func(&self) -> &dyn Fn(&Matrix, &Matrix) -> f32 {
        return &mse
    }
    fn derivative(&self) -> &dyn Fn(&Matrix, &Matrix) -> Matrix {
        return &mse_deriv
    }
}


struct LinearLayer <L: Function> {
    layer_w: Matrix,
    layer_b: Matrix,
    activation_func: L,
    layer_z: Matrix,
    layer_a: Matrix
}

impl<L: Function> LinearLayer<L> {
    fn new(in_size: usize, out_size: usize, activation: L) -> Self{
        return LinearLayer {
            layer_w: Matrix::random(&[out_size, in_size]),
            layer_b: Matrix::random(&[out_size, 1]),
            activation_func: activation,
            layer_z: Matrix::new(&[out_size, 1]),
            layer_a: Matrix::new(&[out_size, 1]),
        }

    }
    fn forward(&mut self, X: &Matrix) -> Matrix {
        self.layer_z = self.layer_w.transpose().matmul(X).add(&self.layer_b);
        self.layer_a = self.layer_z.apply(self.activation_func.func());
        return self.layer_a.clone();
    }
    fn backward(&self, a: &Matrix, error: &Matrix, w: &Matrix, last_layer: bool) -> (Matrix, Matrix, Matrix){
        if !last_layer {
            let deriv = self.layer_z.apply(self.activation_func.derivative());
            let error_l = w.transpose().matmul(error).mul(&deriv);
            let weight_update = error_l.matmul(&a.transpose());
            let bias_update = error_l.clone();
            return (error_l, weight_update, bias_update.clone());
        }
        
        let deriv = self.layer_z.apply(self.activation_func.derivative());
        let error_l = error.mul(&deriv);
        let weight_update = error_l.matmul(&a.transpose());

        let bias_update = error_l.clone();
        return (error_l, weight_update, bias_update.clone());
    }
}

fn main() {
    let matrix = Matrix::from(
        vec![ 
            vec![0.0, 1.0, 0.0],
        ]
    ).transpose();

//    let mut layer = LinearLayer{
//        layer_w: Matrix::random(&[3, 3]),
//        layer_b: Matrix::random(&[1, 3]).transpose(),
//        layer_z: Matrix::new(&[1, 3]).transpose(),
//        activation_func: ReLU{},
//    };
//    let mut layer2 = LinearLayer{
//        layer_w: Matrix::random(&[3, 3]),
//        layer_b: Matrix::random(&[1, 3]).transpose(),
//        layer_z: Matrix::new(&[1, 3]).transpose(),
//        activation_func: Sigmoid{},
//    };
    //
    let mut layer = LinearLayer::new(3, 3, ReLU{});
    let mut layer2 = LinearLayer::new(3, 3, Sigmoid{});


    for i in 0..35 {
        let a1 = layer.forward(&matrix);
        let out = layer2.forward(&a1);
        let mse = MSE{};
        let lr = 0.1;

        let target = Matrix::from(
            vec![ 
                vec![0.0, 1.0, 0.0],
            ]
        ).transpose();
        let loss = mse.func()(&out, &target);
        println!("Loss: {}", loss);

        let error = mse.derivative()(&out, &target);
        let (error_back, weight_update, bias_update) = layer2.backward(&layer.layer_a, &error, &layer2.layer_w, true);
        let (error_back2, weight_update2, bias_update2) = layer.backward(&matrix, &error_back, &layer2.layer_w, false);


        layer2.layer_w = layer2.layer_w.sub(&weight_update.mul_const(1.3));
        layer2.layer_b = layer2.layer_b.sub(&bias_update.mul_const(1.3));

        layer.layer_w = layer.layer_w.sub(&weight_update2.mul_const(1.3));
        layer.layer_b = layer.layer_b.sub(&bias_update2.mul_const(1.3));
    }







}
