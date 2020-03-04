/* eslint-disable require-jsdoc */
import * as tf from '@tensorflow/tfjs';
import * as nd from 'nd4js';
import 'babel-polyfill';

export default function coral(style, content) {
  const flatStyle = tf.reshape(style, [-1, 3]);
  const styleMoments = tf.moments(flatStyle, 0, true);
  const meanStyle = styleMoments.mean;
  const varianceStyle = styleMoments.variance;
  const normStyle = flatStyle.sub(meanStyle).div(tf.sqrt(varianceStyle));
  const flatStyleCovEye =
    tf.matMul(normStyle.transpose(), normStyle) + tf.eye(3);

  const flatContent = tf.reshape(content, [-1, 3]);
  const contentMoments = tf.moments(flatContent, 0, true);
  const meanContent = contentMoments.mean;
  const varianceContent = contentMoments.variance;
  const normContent = flatStyle.sub(meanContent).div(tf.sqrt(varianceContent));
  const flatContentCovEye =
    tf.matMul(normContent.transpose(), normContent).add(tf.eye(3));
  const ndFlat = flatContentCovEye.dataSync();
  matSqrt(ndFlat);
  // styleNormTransfer = tf.matMul(normStyle, tf.matMul(tf.linalg));
}

function matSqrt(m) {
  console.log(m);
  let b = nd.array(m);
  console.log(b);
  const resultSvd = nd.la.svd_decomp(b);
  console.log(resultSvd);
}
// function det and invertMatrix obtained from https://stackoverflow.com/questions/51805389/how-do-i-invert-a-matrix-in-tensorflow-js

function det(m) {
  return tf.tidy(() => {
    const [r, _] = m.shape;
    if (r === 2) {
      const t = m.as1D();
      const a = t.slice([0], [1]).dataSync()[0];
      const b = t.slice([1], [1]).dataSync()[0];
      const c = t.slice([2], [1]).dataSync()[0];
      const d = t.slice([3], [1]).dataSync()[0];
      result = a * d - b * c;
      return result;
    } else {
      let s = 0;
      rows = [...Array(r).keys()];
      for (let i = 0; i < r; i++) {
        sub_m = m.gather(
          tf.tensor1d(
            rows.filter((e) => e !== i),
            'int32'
          )
        );
        sli = sub_m.slice([0, 1], [r - 1, r - 1]);
        s += Math.pow(-1, i) * det(sli);
      }
      return s;
    }
  });
}

// the inverse of the matrix : matrix adjoint method
function invertMatrix(m) {
  return tf.tidy(() => {
    const d = det(m);
    if (d === 0) {
      return;
    }
    [r, _] = m.shape;
    rows = [...Array(r).keys()];
    dets = [];
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < r; j++) {
        const sub_m = m.gather(
          tf.tensor1d(
            rows.filter((e) => e !== i),
            'int32'
          )
        );
        let sli;
        if (j === 0) {
          sli = sub_m.slice([0, 1], [r - 1, r - 1]);
        } else if (j === r - 1) {
          sli = sub_m.slice([0, 0], [r - 1, r - 1]);
        } else {
          const [a, b, c] = tf.split(sub_m, [j, 1, r - (j + 1)], 1);
          sli = tf.concat([a, c], 1);
        }
        dets.push(Math.pow(-1, i + j) * det(sli));
      }
    }
    com = tf.tensor2d(dets, [r, r]);
    tr_com = com.transpose();
    inv_m = tr_com.div(tf.scalar(d));
    return inv_m;
  });
}
