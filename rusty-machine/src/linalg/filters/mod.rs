use super::matrix::{Matrix, MatrixSlice};
use super::utils;

pub fn convolution(w: &Matrix<f64>, mat: &Matrix<f64>) -> Matrix<f64> {
    let mat_rows = mat.rows();
    let mat_cols = mat.cols();

    let conv_rows = w.rows();
    let conv_cols = w.cols();

    // output has size (mat_rows - conv_rows + 1) x (mat_cols - conv_cols + 1)
    let res_rows = mat_rows - conv_rows + 1;
    let res_cols = mat_cols - conv_cols + 1;

    let mut res_data = Vec::with_capacity(res_rows * res_cols);
    for i in 0..res_rows {
        for j in 0..res_cols {
        	
            // Fill the res_data with the convolution
            let mut conv = 0f64;
            let conv_slice = MatrixSlice::from_matrix(mat, [i, j], conv_rows, conv_cols);
            for (m_row, c_row) in conv_slice.iter_rows().zip(w.iter_rows()) {
                conv += utils::dot(m_row, c_row);
            }

            res_data.push(if conv < 0f64 {
                0f64
            } else if conv > 255f64 {
                255f64
            } else {
                conv
            })
        }
    }

    Matrix::new(res_rows, res_cols, res_data)
}

#[cfg(test)]
mod tests {
    use super::convolution;
    use super::super::matrix::Matrix;

    #[test]
    fn test_basic_convolution() {
        let a = Matrix::new(4, 4, vec![2f64; 16]);
        let w = Matrix::new(3, 3, vec![0f64, 0., 0., 0., 1., 0., 0., 0., 0.]);

        let mat = convolution(&w, &a);

        assert_eq!(mat.into_vec(), vec![2.0; 4]);
    }

    #[test]
    fn test_block_convolution() {
        let a = Matrix::new(4, 4, vec![2f64; 16]);
        let w = Matrix::new(3, 3, vec![1f64, 1., 1., 1., 1., 1., 1., 1., 1.]);

        let mat = convolution(&w, &a);

        assert_eq!(mat.into_vec(), vec![18f64; 4]);
    }

    #[test]
    fn test_complicated_convolution() {
        let a = Matrix::new(4, 4, (0..16).map(|x| x as f64).collect::<Vec<_>>());
        let w = Matrix::new(3, 3, vec![4f64, 0., 0., 0., -1., 0., 2., 0., 0.]);

        let mat = convolution(&w, &a);

        assert_eq!(mat.into_vec(), vec![11f64, 16f64, 31f64, 36f64]);
    }

    #[test]
    fn test_small_convolution() {
        let a = Matrix::new(4, 4, (0..16).map(|x| x as f64).collect::<Vec<_>>());
        let w = Matrix::new(2, 2, vec![0., -1.0, 0., 1.0]);

        let mat = convolution(&w, &a);

        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.rows(), 3);

        assert_eq!(mat.into_vec(), vec![4.0; 9]);
    }
}
