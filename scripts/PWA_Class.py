import pandas as pd
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from functools import partial
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union
from pathlib import Path

# Sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.utils import resample

# Other ML
from lightgbm import LGBMRegressor

# tensorflow
import tensorflow as tf
import tensorflow.compat.v2.summary as summary
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D, Dense, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.python.ops import summary_ops_v2
from tensorflow.keras.utils import plot_model

# pytorch
import torch
import torchtuples as tt

# survival
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.models import CoxPH

# Hyperparameters
from hyperopt import hp, tpe, fmin, Trials


# Helper classes and functions for deep survival, from:
# https://colab.research.google.com/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb#scrollTo=azrczYYVvEQb
class TrainAndEvaluateModel:
    
    def __init__(self, model, train_dataset, train_predonly_dataset, eval_dataset, test_dataset, learning_rate,
                 num_epochs):
        self.num_epochs = num_epochs
        self.model = model
        self.train_ds = train_dataset
        self.train_predonly_ds = train_predonly_dataset
        self.val_ds = eval_dataset
        self.test_ds = test_dataset
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.train_cindex_metric = CindexMetric()
        self.val_cindex_metric = CindexMetric()
        self.test_cindex_metric = CindexMetric()
    
    @tf.function
    def train_one_step(self, x_cnn, x_side, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        with tf.GradientTape() as tape:
            logits = self.model([x_cnn, x_side], training=True)
            train_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=logits)
        
        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss, logits
    
    def train_and_evaluate(self):
        step = tf.Variable(0, dtype=tf.int64)
        for epoch in range(self.num_epochs):
            self.train_one_epoch(step)
            # Run a validation loop at the end of each epoch.
            self.evaluate(step)
    
    def train_one_epoch(self, step_counter):
        for x, y in self.train_ds:
            train_loss, logits = self.train_one_step(
                x["input_cnn"], x["input_side"], y["label_event"], y["label_riskset"])
            
            step = int(step_counter)
            if step == 0:
                # see https://stackoverflow.com/questions/58843269/display-graph-using-tensorflow-v2-0-in-tensorboard
                func = self.train_one_step.get_concrete_function(
                    x["input_cnn"], x["input_side"], y["label_event"], y["label_riskset"])
                summary_ops_v2.graph(func.graph, step=0)
            
            # Update training metric.
            self.train_loss_metric.update_state(train_loss)
            
            # Log every 200 batches.
            if step % 200 == 0:
                # Display metrics
                mean_loss = self.train_loss_metric.result()
                print(f"step {step}: mean loss = {mean_loss:.4f}")
                # save summaries
                summary.scalar("loss", mean_loss, step=step_counter)
                # Reset training metrics
                self.train_loss_metric.reset_states()
            
            step_counter.assign_add(1)
    
    @tf.function
    def evaluate_one_step(self, x_cnn, x_side, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        val_logits = self.model([x_cnn, x_side], training=False)
        val_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
        return val_loss, val_logits
    
    def evaluate(self, step_counter):
        self.val_cindex_metric.reset_states()
        
        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(
                x_val["input_cnn"], x_val["input_side"], y_val["label_event"], y_val["label_riskset"])
            
            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_cindex_metric.update_state(y_val, val_logits)
        
        val_loss = self.val_loss_metric.result()
        summary.scalar("loss",
                       val_loss,
                       step=step_counter)
        self.val_loss_metric.reset_states()
        
        val_cindex = self.val_cindex_metric.result()
        for key, value in val_cindex.items():
            summary.scalar(key, value, step=step_counter)
        
        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")
    
    def generate_predictions(self):
        self.train_cindex_metric.reset_states()
        self.val_cindex_metric.reset_states()
        self.test_cindex_metric.reset_states()
        
        for x_train, y_train in self.train_predonly_ds:
            train_loss, train_logits = self.evaluate_one_step(
                x_train["input_cnn"], x_train["input_side"], y_train["label_event"], y_train["label_riskset"])
            self.train_cindex_metric.update_state(y_train, train_logits)
        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(
                x_val["input_cnn"], x_val["input_side"], y_val["label_event"], y_val["label_riskset"])
            self.val_cindex_metric.update_state(y_val, val_logits)
        for x_test, y_test in self.test_ds:
            test_loss, test_logits = self.evaluate_one_step(
                x_test["input_cnn"], x_test["input_side"], y_test["label_event"], y_test["label_riskset"])
            self.test_cindex_metric.update_state(y_test, test_logits)
        
        predictions = {
            'train': self.train_cindex_metric.generate_predictions()['prediction'],
            'val': self.val_cindex_metric.generate_predictions()['prediction'],
            'test': self.test_cindex_metric.generate_predictions()['prediction']
        }
        return predictions

def _make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set


class InputFunction:
    """Callable input function that computes the risk set for each batch.

    Parameters
    ----------
    features: np.ndarray, shape=(n_samples, n_features)
    images : np.ndarray, shape=(n_samples, height, width)
        Image data.
    time : np.ndarray, shape=(n_samples,)
        Observed time.
    event : np.ndarray, shape=(n_samples,)
        Event indicator.
    batch_size : int, optional, default=64
        Number of samples per batch.
    drop_last : int, optional, default=False
        Whether to drop the last incomplete batch.
    shuffle : bool, optional, default=False
        Whether to shuffle data.
    seed : int, optional, default=89
        Random number seed.
    """
    
    def __init__(self,
                 images: np.ndarray,
                 features: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray,
                 batch_size: int = 16384,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 seed: int = 89) -> None:
        if images.ndim == 3:
            images = images[..., np.newaxis]
        self.images = images
        self.features = features
        self.time = time
        self.event = event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
    
    def size(self) -> int:
        """Total number of samples."""
        return self.images.shape[0]
    
    def steps_per_epoch(self) -> int:
        """Number of batches for one epoch."""
        return int(np.floor(self.size() / self.batch_size))
    
    def _get_data_batch(self, index: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute risk set for samples in batch."""
        time = self.time[index]
        event = self.event[index]
        images = self.images[index]
        features = self.features.values[index]
        inputs = {
            "input_cnn": images.astype(np.float32),
            "input_side": features.astype(np.float32)
        }
        labels = {
            "label_event": event.astype(np.int32),
            "label_time": time.astype(np.float32),
            "label_riskset": _make_riskset(time)
        }
        return inputs, labels
    
    def _iter_data(self) -> Iterable[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Generator that yields one batch at a time."""
        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)
        
        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch_size)]
            yield self._get_data_batch(idx)
        
        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            yield self._get_data_batch(idx)
    
    def _get_shapes(self) -> Tuple[Dict[str, tf.TensorShape], Dict[str, tf.TensorShape]]:
        """Return shapes of data returned by `self._iter_data`."""
        batch_size = self.batch_size if self.drop_last else None
        l, w, c = self.images.shape[1:]
        images = tf.TensorShape([batch_size, l, w, c])
        f = self.features.shape[1]
        features = tf.TensorShape([batch_size, f])
        inputs = {"input_cnn": images, "input_side": features}
        labels = {k: tf.TensorShape((batch_size,))
                  for k in ("label_event", "label_time")}
        labels["label_riskset"] = tf.TensorShape((batch_size, batch_size))
        return inputs, labels
    
    def _get_dtypes(self) -> Tuple[Dict[str, tf.DType], Dict[str, tf.DType]]:
        """Return dtypes of data returned by `self._iter_data`."""
        inputs = {"input_cnn": tf.float32,
                  "input_side": tf.float32}
        labels = {"label_event": tf.int32,
                  "label_time": tf.float32,
                  "label_riskset": tf.bool}
        return inputs, labels
    
    def _make_dataset(self) -> tf.data.Dataset:
        """Create dataset from generator."""
        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            self._get_dtypes(),
            self._get_shapes()
        )
        return ds
    
    def __call__(self) -> tf.data.Dataset:
        return self._make_dataset()

def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm

def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)
        # for numerical stability, substract the maximum value
        # before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output


class CoxPHLoss(tf.keras.losses.Loss):
    """Negative partial log-likelihood of Cox's proportional hazards model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,
             y_true: Sequence[tf.Tensor],
             y_pred: tf.Tensor) -> tf.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_true : list|tuple of tf.Tensor
            The first element holds a binary vector where 1
            indicates an event 0 censoring.
            The second element holds the riskset, a
            boolean matrix where the `i`-th row denotes the
            risk set of the `i`-th instance, i.e. the indices `j`
            for which the observer time `y_j >= y_i`.
            Both must be rank 2 tensors.
        y_pred : tf.Tensor
            The predicted outputs. Must be a rank 2 tensor.

        Returns
        -------
        loss : tf.Tensor
            Loss for each instance in the batch.
        """
        event, riskset = y_true
        predictions = y_pred

        pred_shape = predictions.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)

        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal rank of event (received %s)" % (
                pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)

        event = tf.cast(event, predictions.dtype)
        predictions = safe_normalize(predictions)

        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        pred_t = tf.transpose(predictions)
        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()

        losses = tf.math.multiply(event, rr - predictions)

        return losses


class CindexMetric:
    """Computes concordance index across one epoch."""

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def update_state(self, y_true: Dict[str, tf.Tensor], y_pred: tf.Tensor) -> None:
        """Collect observed time, event indicator and predictions for a batch.

        Parameters
        ----------
        y_true : dict
            Must have two items:
            `label_time`, a tensor containing observed time for one batch,
            and `label_event`, a tensor containing event indicator for one batch.
        y_pred : tf.Tensor
            Tensor containing predicted risk score for one batch.
        """
        self._data["label_time"].append(y_true["label_time"].numpy())
        self._data["label_event"].append(y_true["label_event"].numpy())
        self._data["prediction"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        results = concordance_index(
            data["label_time"],
            -data["prediction"],
            data["label_event"] == 1
            )

        result_data = {}
        result_data["cindex"] = results

        return result_data
    
    def generate_predictions(self):
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)
        return data
    


# Custom classes
class Basics:
    
    def __init__(self):
        # seeds for reproducibility
        self.seed = 0
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)
        _ = torch.manual_seed(self.seed)
        
        # other parameters
        self.path_data = '../data/'
        self.folds = ['train', 'val', 'test']
        self.modes = ['', '_sd', '_str']
        self.id_vars = ['id', 'eid', 'instance']
        self.instances = ['0', '1', '2', '3']
        self.ethnicities_vars = \
            ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other', 'Ethnicity.Mixed',
             'Ethnicity.White_and_Black_Caribbean', 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
             'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian', 'Ethnicity.Pakistani',
             'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other', 'Ethnicity.Black', 'Ethnicity.Caribbean',
             'Ethnicity.African', 'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other',
             'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
        self.demographic_vars = ['Age', 'Sex'] + self.ethnicities_vars
        self.survival_vars = ['FollowUpTime', 'Death']
        self.biological_vars = ['Absence of notch position in the pulse waveform', 'Position of pulse wave notch',
                                'Position of the pulse wave peak', 'Position of the shoulder on the pulse waveform',
                                'Pulse rate',
                                'Pulse wave Arterial Stiffness index', 'Pulse wave peak to peak time',
                                'Pulse wave reflection index']
        self.features = self.demographic_vars + self.survival_vars + self.biological_vars


class Preprocessing(Basics):
    
    def __init__(self):
        Basics.__init__(self)
        self.data_raw = None
        self.data_features = None
        self.data_features_eids = None
    
    def _preprocessing(self):
        dict_UKB_fields_to_names = {'34-0.0': 'Year_of_birth', '52-0.0': 'Month_of_birth',
                                    '53-0.0': 'Date_attended_center_0', '53-1.0': 'Date_attended_center_1',
                                    '53-2.0': 'Date_attended_center_2', '53-3.0': 'Date_attended_center_3',
                                    '40000-0.0': 'FollowUpDate', '31-0.0': 'Sex', '22001-0.0': 'Sex_genetic',
                                    '21000-0.0': 'Ethnicity', '21000-1.0': 'Ethnicity_1', '21000-2.0': 'Ethnicity_2'}
        for instance in self.instances:
            dict_UKB_fields_to_names['4205-' + instance + '.0'] = 'PWA_' + instance
            dict_UKB_fields_to_names[
                '4204-' + instance + '.0'] = 'Absence of notch position in the pulse waveform_' + instance
            dict_UKB_fields_to_names['4199-' + instance + '.0'] = 'Position of pulse wave notch_' + instance
            dict_UKB_fields_to_names['4198-' + instance + '.0'] = 'Position of the pulse wave peak_' + instance
            dict_UKB_fields_to_names[
                '4200-' + instance + '.0'] = 'Position of the shoulder on the pulse waveform_' + instance
            dict_UKB_fields_to_names['4194-' + instance + '.0'] = 'Pulse rate_' + instance
            dict_UKB_fields_to_names['21021-' + instance + '.0'] = 'Pulse wave Arterial Stiffness index_' + instance
            dict_UKB_fields_to_names['4196-' + instance + '.0'] = 'Pulse wave peak to peak time_' + instance
            dict_UKB_fields_to_names['4195-' + instance + '.0'] = 'Pulse wave reflection index_' + instance
        
        self.data_raw.rename(columns=dict_UKB_fields_to_names, inplace=True)
        self.data_raw['eid'] = self.data_raw['eid'].astype(str)
        self.data_raw.set_index('eid', drop=False, inplace=True)
        self.data_raw.index.name = 'column_names'
        self.data_raw.dropna(how='all', subset=['PWA_0', 'PWA_1', 'PWA_2', 'PWA_3'], inplace=True)
        # Format survival data
        self.data_raw['Death'] = ~self.data_raw['FollowUpDate'].isna()
        self.data_raw['FollowUpDate'][self.data_raw['FollowUpDate'].isna()] = '2020-04-27'
        self.data_raw['FollowUpDate'] = self.data_raw['FollowUpDate'].apply(
            lambda x: pd.NaT if pd.isna(x) else datetime.strptime(x, '%Y-%m-%d'))
        assert ('FollowUpDate.1' not in self.data_raw.columns)
    
    def _compute_sex(self):
        # Use genetic sex when available
        self.data_raw['Sex_genetic'][self.data_raw['Sex_genetic'].isna()] = \
            self.data_raw['Sex'][self.data_raw['Sex_genetic'].isna()]
        self.data_raw.drop(['Sex'], axis=1, inplace=True)
        self.data_raw.rename(columns={'Sex_genetic': 'Sex'}, inplace=True)
        self.data_raw = self.data_raw.dropna(subset=['Sex'])
    
    def _compute_age(self):
        # Recompute age with greater precision by leveraging the month of birth
        self.data_raw['Year_of_birth'] = self.data_raw['Year_of_birth'].astype(int)
        self.data_raw['Month_of_birth'] = self.data_raw['Month_of_birth'].astype(int)
        self.data_raw['Date_of_birth'] = self.data_raw.apply(
            lambda row: datetime(row.Year_of_birth, row.Month_of_birth, 15), axis=1)
        for i in self.instances:
            self.data_raw['Date_attended_center_' + i] = \
                self.data_raw['Date_attended_center_' + i].apply(
                    lambda x: pd.NaT if pd.isna(x) else datetime.strptime(x, '%Y-%m-%d'))
            self.data_raw['Age_' + i] = self.data_raw['Date_attended_center_' + i] - self.data_raw['Date_of_birth']
            self.data_raw['Age_' + i] = self.data_raw['Age_' + i].dt.days / 365.25
            self.data_raw['FollowUpTime_' + i] = self.data_raw['FollowUpDate'] - self.data_raw[
                'Date_attended_center_' + i]
            self.data_raw['FollowUpTime_' + i] = self.data_raw['FollowUpTime_' + i].dt.days / 365.25
            self.data_raw.drop(['Date_attended_center_' + i], axis=1, inplace=True)
        self.data_raw.drop(['Year_of_birth', 'Month_of_birth', 'Date_of_birth', 'FollowUpDate'], axis=1, inplace=True)
        self.data_raw.dropna(how='all', subset=['Age_0', 'Age_1', 'Age_2', 'Age_3'], inplace=True)
        assert ('FollowUpDate.1' not in self.data_raw.columns)
    
    def _encode_ethnicity(self):
        # Fill NAs for ethnicity on instance 0 if available in other instances
        eids_missing_ethnicity = self.data_raw['eid'][self.data_raw['Ethnicity'].isna()]
        for eid in eids_missing_ethnicity:
            sample = self.data_raw.loc[eid, :]
            if not math.isnan(sample['Ethnicity_1']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_1']
            elif not math.isnan(sample['Ethnicity_2']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_2']
        self.data_raw.drop(['Ethnicity_1', 'Ethnicity_2'], axis=1, inplace=True)
        
        # One hot encode ethnicity
        dict_ethnicity_codes = {'1': 'Ethnicity.White', '1001': 'Ethnicity.British', '1002': 'Ethnicity.Irish',
                                '1003': 'Ethnicity.White_Other',
                                '2': 'Ethnicity.Mixed', '2001': 'Ethnicity.White_and_Black_Caribbean',
                                '2002': 'Ethnicity.White_and_Black_African',
                                '2003': 'Ethnicity.White_and_Asian', '2004': 'Ethnicity.Mixed_Other',
                                '3': 'Ethnicity.Asian', '3001': 'Ethnicity.Indian', '3002': 'Ethnicity.Pakistani',
                                '3003': 'Ethnicity.Bangladeshi', '3004': 'Ethnicity.Asian_Other',
                                '4': 'Ethnicity.Black', '4001': 'Ethnicity.Caribbean', '4002': 'Ethnicity.African',
                                '4003': 'Ethnicity.Black_Other',
                                '5': 'Ethnicity.Chinese',
                                '6': 'Ethnicity.Other_ethnicity',
                                '-1': 'Ethnicity.Do_not_know',
                                '-3': 'Ethnicity.Prefer_not_to_answer',
                                '-5': 'Ethnicity.NA'}
        self.data_raw['Ethnicity'] = self.data_raw['Ethnicity'].fillna(-5).astype(int).astype(str)
        ethnicities = pd.get_dummies(self.data_raw['Ethnicity'])
        self.data_raw.drop(['Ethnicity'], axis=1, inplace=True)
        ethnicities.rename(columns=dict_ethnicity_codes, inplace=True)
        ethnicities['Ethnicity.White'] = ethnicities['Ethnicity.White'] + ethnicities['Ethnicity.British'] + \
                                         ethnicities['Ethnicity.Irish'] + ethnicities['Ethnicity.White_Other']
        ethnicities['Ethnicity.Mixed'] = ethnicities['Ethnicity.Mixed'] + \
                                         ethnicities['Ethnicity.White_and_Black_Caribbean'] + \
                                         ethnicities['Ethnicity.White_and_Black_African'] + \
                                         ethnicities['Ethnicity.White_and_Asian'] + \
                                         ethnicities['Ethnicity.Mixed_Other']
        ethnicities['Ethnicity.Asian'] = ethnicities['Ethnicity.Asian'] + ethnicities['Ethnicity.Indian'] + \
                                         ethnicities['Ethnicity.Pakistani'] + ethnicities['Ethnicity.Bangladeshi'] + \
                                         ethnicities['Ethnicity.Asian_Other']
        ethnicities['Ethnicity.Black'] = ethnicities['Ethnicity.Black'] + ethnicities['Ethnicity.Caribbean'] + \
                                         ethnicities['Ethnicity.African'] + ethnicities['Ethnicity.Black_Other']
        ethnicities['Ethnicity.Other'] = ethnicities['Ethnicity.Other_ethnicity'] + \
                                         ethnicities['Ethnicity.Do_not_know'] + \
                                         ethnicities['Ethnicity.Prefer_not_to_answer'] + \
                                         ethnicities['Ethnicity.NA']
        self.data_raw = self.data_raw.join(ethnicities)
    
    def _concatenate_instances(self):
        self.data_features = None
        for i in self.instances:
            print('Preparing the samples for instance ' + i)
            df_i = self.data_raw.dropna(subset=['Age_' + i])
            print(str(len(df_i.index)) + ' samples found in instance ' + i)
            dict_names = {}
            features = ['Age', 'FollowUpTime', 'PWA'] + self.biological_vars
            
            for feature in features:
                dict_names[feature + '_' + i] = feature
            self.dict_names = dict_names
            df_i.rename(columns=dict_names, inplace=True)
            df_i['instance'] = i
            df_i['id'] = df_i['eid'] + '_' + df_i['instance']
            df_i = df_i[self.id_vars + self.survival_vars + self.demographic_vars + features[2:]]
            if self.data_features is None:
                self.data_features = df_i
            else:
                self.data_features = self.data_features.append(df_i)
            print('The size of the full concatenated dataframe is now ' + str(len(self.data_features.index)))
    
    def _postprocessing(self):
        # Save age as a float32 instead of float64
        self.data_features['Age'] = np.float32(self.data_features['Age'])
        # Convert the Death event to an int
        self.data_features['Death'] = self.data_features['Death'].astype(int)
        # Drop the NAs
        self.data_features.dropna(inplace=True)
        # Set ID as index
        self.data_features.set_index('id', inplace=True)
        # Shuffle the rows before saving the dataframe
        self.data_features = self.data_features.sample(frac=1)
    
    def _extract_pwa(self):
        def preprocess(row):
            pairs = row['PWA'].split('|')[1:-1]
            time_serie = pd.Series([row['eid']] + [int(pair.split(',')[1]) for pair in pairs])
            return time_serie
        
        self.pwa = self.data_features.apply(preprocess, axis=1)
        self.pwa.columns = ['eid'] + ['t' + str(i) for i in range(100)]
        self.data_features.drop(columns=['PWA'], inplace=True)
    
    def _split_data(self):
        # Generate IDs split
        EIDS = {}
        eids = self.data_features['eid'].unique()
        n_train = int(len(eids) * 0.8)
        n_train_val = int(len(eids) * 0.9)
        EIDS['train'] = eids[:n_train]
        EIDS['val'] = eids[n_train:n_train_val]
        EIDS['test'] = eids[n_train_val:]
        self.EIDS = EIDS
        
        # Split data features and PWA
        self.DATA_FEATURES = {}
        self.PWA = {}
        for fold in self.folds:
            self.DATA_FEATURES[fold] = self.data_features[self.data_features['eid'].isin(EIDS[fold])]
            self.PWA[fold] = self.pwa[self.pwa['eid'].isin(EIDS[fold])]
            self.PWA[fold].drop(columns=['eid'], inplace=True)
    
    def _normalize_datasets(self):
        # Compute statistics on training set to normalize all sets
        max_pwa_train = self.PWA['train'].max().max()
        scalar_variables = [col for col in self.DATA_FEATURES['train'].columns.values if
                            col not in ['eid', 'instance', 'FollowUpTime', 'Death']]
        scalar_variables_mean_train = self.DATA_FEATURES['train'][scalar_variables].mean()
        scalar_variables_std_train = self.DATA_FEATURES['train'][scalar_variables].std()
        for fold in self.folds:
            self.PWA[fold] /= max_pwa_train
            self.DATA_FEATURES[fold].loc[:, scalar_variables] -= scalar_variables_mean_train
            self.DATA_FEATURES[fold].loc[:, scalar_variables] /= scalar_variables_std_train
    
    def generate_data(self):
        # Preprocessing
        usecols = ['eid', '31-0.0', '22001-0.0', '21000-0.0', '21000-1.0', '21000-2.0', '40000-0.0', '34-0.0', '52-0.0',
                   '53-0.0',
                   '53-1.0', '53-2.0', '53-3.0']
        for instance in self.instances:
            usecols.extend(['4205-' + instance + '.0', '4204-' + instance + '.0', '4199-' + instance + '.0',
                            '4198-' + instance + '.0', '4200-' + instance + '.0', '4194-' + instance + '.0',
                            '21021-' + instance + '.0',
                            '4196-' + instance + '.0', '4195-' + instance + '.0'])
        
        self.data_raw = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv', usecols=usecols)
        
        # Formatting
        self._preprocessing()
        self._compute_sex()
        self._compute_age()
        self._encode_ethnicity()
        self._concatenate_instances()
        self._postprocessing()
        self._extract_pwa()
        self._split_data()
        self._normalize_datasets()
    
    def save_data(self):
        for fold in self.folds:
            self.DATA_FEATURES[fold].to_csv('../data/data_features_' + fold + '.csv')
            self.PWA[fold].to_csv('../data/PWA_' + fold + '.csv')


class Predictions(Basics):
    
    def __init__(self, target=None, predictors=None, algo_name=None):
        Basics.__init__(self)
        self.target = target
        self.predictors = predictors
        self.algo_name = algo_name
        if self.algo_name == 'CNN':
            self.path_weights = '../data/CNN-weights_' + self.target + '_' + self.predictors + '.h5'
        else:
            self.path_weights = None
        if self.target == 'Age':
            self.algorithms = {'ElasticNet': ElasticNet, 'GBM': LGBMRegressor, 'NeuralNetwork': MLPRegressor}
            self.n_hyperopt_evals = {'ElasticNet': 200, 'GBM': 200, 'NeuralNetwork': 15, 'CNN': 30}
        elif self.target == 'Survival':
            self.algorithms = {'ElasticNet': CoxPHFitter, 'GBM': GradientBoostingSurvivalAnalysis,
                               'NeuralNetwork': tt.practical.MLPVanilla}
            self.n_hyperopt_evals = {'ElasticNet': 50, 'GBM': 10, 'NeuralNetwork': 10, 'CNN': 10}
        else:
            print('This target was not coded.')
            sys.exit(1)
        self.DATA_FEATUREs = {}
        self.PWAs = {}
        self.Xs = {}
        self.Ys = {}
        self.SPACES = None
        self.best_hyperparameters = None
        self.model = None
        self.trainer = None
        self.Feature_importance = None
        self.PREDs = {}
        self.PERFs = {}
        self.parameters_int = ['num_leaves', 'min_child_samples', 'n_estimators']
        
        # SPACES
        self.SPACES = {
            'Age': {
                'ElasticNet': {
                    'alpha': hp.loguniform('alpha', -10, 0),
                    'l1_ratio': hp.uniform('l1_ratio', 0, 1)
                },
                'GBM': {
                    'num_leaves': hp.quniform('num_leaves', low=5, high=45, q=1),
                    'min_child_samples': hp.quniform('min_child_samples', low=100, high=500, q=1),
                    'min_child_weight': hp.loguniform('min_child_weight', low=-5, high=4),
                    'subsample': hp.uniform('subsample', low=0.2, high=0.8),
                    'colsample_bytree': hp.uniform('colsample_bytree', low=0.4, high=0.6),
                    'reg_alpha': hp.loguniform('reg_alpha', low=-2, high=2),
                    'reg_lambda': hp.loguniform('reg_lambda', low=-2, high=2),
                    'n_estimators': hp.quniform('n_estimators', low=150, high=450, q=1)
                },
                'NeuralNetwork': {
                    'learning_rate_init': hp.loguniform('learning_rate_init', low=-5, high=-1),
                    'alpha': hp.loguniform('alpha', low=-6, high=-1)
                },
                'CNN': {
                    'learning_rate': hp.loguniform('learning_rate', low=-5, high=-3),
                    'weight_decay': hp.loguniform('weight_decay', low=-5, high=3),
                    'dropout_rate': hp.uniform('dropout_rate', low=0, high=0.9),
                    'CNN_BatchNormalization': hp.choice('CNN_BatchNormalization', [1, 0]),
                    'DNN_BatchNormalization': hp.choice('DNN_BatchNormalization', [1, 0])
                }
            },
            'Survival': {
                'ElasticNet': {
                    'penalizer': hp.loguniform('penalizer', -10, 0),
                    'l1_ratio': hp.uniform('l1_ratio', 0, 1)
                },
                'CNN': {
                    'learning_rate': hp.loguniform('learning_rate', low=-5, high=-3),
                    'weight_decay': hp.loguniform('weight_decay', low=-5, high=3),
                    'dropout_rate': hp.uniform('dropout_rate', low=0, high=0.9),
                    'CNN_BatchNormalization': hp.choice('CNN_BatchNormalization', [True, False]),
                    'DNN_BatchNormalization': hp.choice('DNN_BatchNormalization', [True, False])
                }
            }
        }
    
    def preprocessing(self):
        for fold in self.folds:
            # Predictors
            self.DATA_FEATUREs[fold] = pd.read_csv('../data/data_features_' + fold + '.csv', index_col=0)
            if self.predictors == 'demographics':
                self.Xs[fold] = self.DATA_FEATUREs[fold][self.demographic_vars]
            elif self.predictors == 'features':
                self.Xs[fold] = self.DATA_FEATUREs[fold][self.demographic_vars + self.biological_vars]
            elif self.predictors == 'PWA':
                data_features = self.DATA_FEATUREs[fold][self.demographic_vars]
                data_pwa = pd.read_csv('../data/PWA_' + fold + '.csv', index_col=0)
                self.Xs[fold] = data_features.merge(data_pwa, right_index=True, left_index=True)
            elif self.predictors == 'all':
                data_features = self.DATA_FEATUREs[fold][self.demographic_vars + self.biological_vars]
                data_pwa = pd.read_csv('../data/PWA_' + fold + '.csv', index_col=0)
                self.Xs[fold] = data_features.merge(data_pwa, right_index=True, left_index=True)
            elif self.predictors == 'demographics+PWA':
                data_features = self.DATA_FEATUREs[fold][self.demographic_vars]
                data_pwa = pd.read_csv('../data/PWA_' + fold + '.csv', index_col=0)
                data_pwa = np.expand_dims(data_pwa.values, axis=2)
                self.Xs[fold] = [data_pwa, data_features]
            elif self.predictors == 'features+PWA':
                data_features = self.DATA_FEATUREs[fold][self.demographic_vars + self.biological_vars]
                data_pwa = pd.read_csv('../data/PWA_' + fold + '.csv', index_col=0)
                data_pwa = np.expand_dims(data_pwa.values, axis=2)
                self.Xs[fold] = [data_pwa, data_features]
            elif self.predictors == 'all+PWA':
                data_features = self.DATA_FEATUREs[fold][self.demographic_vars + self.biological_vars]
                data_pwa = pd.read_csv('../data/PWA_' + fold + '.csv', index_col=0)
                data_features = data_features.merge(data_pwa, right_index=True, left_index=True)
                data_pwa = np.expand_dims(data_pwa.values, axis=2)
                self.Xs[fold] = [data_pwa, data_features]
            else:
                print('This set of predictors was not coded.')
                sys.exit(1)
            
            # Target related
            if self.target == 'Age':
                self.Ys[fold] = self.DATA_FEATUREs[fold]['Age']
                if self.predictors in ['demographics+PWA', 'features+PWA', 'all+PWA']:
                    self.Xs[fold][1].drop(columns=['Age'], inplace=True)
                else:
                    self.Xs[fold].drop(columns=['Age'], inplace=True)
            elif self.target == 'Survival':
                self.Ys[fold] = self.DATA_FEATUREs[fold][['FollowUpTime', 'Death']]
                self.PREDs[fold] = self.Ys[fold].copy()
                if self.algo_name == 'ElasticNet':
                    self.Xs[fold].drop(columns=['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Mixed',
                                                'Ethnicity.Asian', 'Ethnicity.Black', 'Ethnicity.Chinese',
                                                'Ethnicity.Other'], inplace=True)
                elif self.algo_name == 'GBM':
                    self.Ys[fold] = Surv.from_arrays(self.Ys[fold]['Death'], self.Ys[fold]['FollowUpTime'])
                elif self.algo_name == 'NeuralNetwork':
                    self.Ys[fold] = (self.Ys[fold]['FollowUpTime'].astype('float32').values,
                                     self.Ys[fold]['Death'].astype('int32').values)
                    self.Xs[fold] = self.Xs[fold].astype('float32').values
    
    def _generate_age_CNN(self, learning_rate, weight_decay, dropout_rate, CNN_BatchNormalization,
                          DNN_BatchNormalization):
        # Convolutions
        CNN_input = Input(shape=(100, 1))
        CNN = CNN_input
        for filters in [32, 64, 128, 256]:
            CNN = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation='relu')(CNN)
            if CNN_BatchNormalization:
                CNN = BatchNormalization()(CNN)
            CNN = MaxPooling1D(pool_size=2)(CNN)
        CNN = GlobalMaxPooling1D()(CNN)
        # Side Dense Neural Network (SDNN)
        if self.predictors == 'demographics+PWA':
            SDNN_input = Input(shape=(25))
            SDNN = Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(SDNN_input)
        elif self.predictors == 'features+PWA':
            SDNN_input = Input(shape=(33))
            SDNN = Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(SDNN_input)
        elif self.predictors == 'all+PWA':
            SDNN_input = Input(shape=(133))
            SDNN = Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(SDNN_input)
        else:
            print('ERROR: CNN is only available for predictors demographics+PWA, features+PWA and all+PWA')
            sys.exit(1)
        # Dense layers
        DNN = concatenate([CNN, SDNN])
        for units in [256, 128, 64, 32]:
            DNN = Dense(units=units, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(DNN)
            if DNN_BatchNormalization:
                DNN = BatchNormalization()(DNN)
            DNN = Dropout(dropout_rate)(DNN)
        # Final layer for regression prediction
        DNN = Dense(units=1, activation='linear')(DNN)
        model = Model(inputs=[CNN_input, SDNN_input], outputs=DNN)
        # Compiler
        model.compile(loss='MSE', optimizer=Adam(lr=learning_rate, clipnorm=1.0),
                           metrics=[RootMeanSquaredError(name='RMSE')])
        return model
    
    def _generate_surv_CNN(self, weight_decay, dropout_rate, CNN_BatchNormalization, DNN_BatchNormalization):
        # Convolutions
        CNN_input = Input(shape=(100, 1, 1))
        CNN = CNN_input
        for filters in [32, 64, 128, 256]:
            CNN = Conv2D(filters=filters, kernel_size=(3, 1), strides=(1, 1), padding="same", activation='relu')(CNN)
            if CNN_BatchNormalization:
                CNN = BatchNormalization()(CNN)
            CNN = MaxPooling2D(pool_size=(2, 1))(CNN)
        CNN = GlobalMaxPooling2D()(CNN)
        # Side Dense Neural Network (SDNN)
        if self.predictors == 'demographics+PWA':
            SDNN_input = Input(shape=(26))
            SDNN = Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(SDNN_input)
        elif self.predictors == 'features+PWA':
            SDNN_input = Input(shape=(34))
            SDNN = Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(SDNN_input)
        elif self.predictors == 'all+PWA':
            SDNN_input = Input(shape=(134))
            SDNN = Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(SDNN_input)
        else:
            print('ERROR: CNN is only available for predictors features+PWA and all+PWA')
            sys.exit(1)
        # Dense layers
        DNN = concatenate([CNN, SDNN])
        for units in [256, 128, 64, 32]:
            DNN = Dense(units=units, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(DNN)
            if DNN_BatchNormalization:
                DNN = BatchNormalization()(DNN)
            DNN = Dropout(dropout_rate)(DNN)
        # Final layer for regression prediction
        DNN = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(weight_decay))(DNN)
        model = Model(inputs=[CNN_input, SDNN_input], outputs=DNN)
        return model
    
    def _objective(self, params=None, algo_name=None):
        print('Hyperopt: testing the following hyperparameters combination:')
        print(params)
        if self.target == 'Age':
            # format as int if needed
            for parameter in self.parameters_int:
                if parameter in params.keys():
                    params[parameter] = int(params[parameter])
            if algo_name == 'NeuralNetwork':
                params.update(
                    {'solver': 'adam', 'activation': 'relu', 'hidden_layer_sizes': (256, 128, 64, 32),
                     'batch_size': 16384, 'shuffle': True, 'max_iter': 10000, 'early_stopping': True,
                     'n_iter_no_change': 50, 'verbose': 1})
            if algo_name == 'CNN':
                model = self._generate_age_CNN(learning_rate=params['learning_rate'],
                                               weight_decay=params['weight_decay'],
                                               dropout_rate=params['dropout_rate'],
                                               CNN_BatchNormalization=params['CNN_BatchNormalization'],
                                               DNN_BatchNormalization=params['DNN_BatchNormalization'])
                early_stopping = EarlyStopping(monitor='val_RMSE', min_delta=0, patience=100, verbose=0, mode='min',
                                               restore_best_weights=True)
                reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='min',
                                                         min_delta=0.0)
                model.fit(self.Xs['train'], self.Ys['train'], validation_data=(self.Xs['val'], self.Ys['val']),
                          epochs=1000, batch_size=32768, shuffle=True, callbacks=[early_stopping, reduce_lr_on_plateau],
                          verbose=0)
            else:
                algo = self.algorithms[algo_name]
                model = algo(**params)
                model.fit(self.Xs['train'], self.Ys['train'])
            pred_val = model.predict(self.Xs['val'])
            r2 = r2_score(self.Ys['val'], pred_val)
            print('R2_val=' + str(round(r2, 3)))
            score = 1 - r2
        elif self.target == 'Survival':
            if self.algo_name == 'CNN':
                print('No hyperparameters tuning, relying on the tuning performed for age prediction.')
            elif self.algo_name == 'ElasticNet':
                df_train = self.Ys['train'].merge(self.Xs['train'], right_index=True, left_index=True)
                algo = self.algorithms[algo_name]
                model = algo(**params)
                model.fit(df_train, duration_col='FollowUpTime', event_col='Death')
                df_val = self.Ys['val'].merge(self.Xs['val'], right_index=True, left_index=True)
            # Predictions
            if self.algo_name == 'ElasticNet':
                pred = model.predict_partial_hazard(df_val)
            else:
                pred = model.predict(df_val)
            # Performance
            ci = concordance_index(df_val['FollowUpTime'], -pred, df_val['Death'])
            print('CI_val = ' + str(round(ci, 3)))
            score = 1 - ci
        return score
    
    def hyperparameters_tuning(self):
        if self.target == 'Survival' and self.algo_name == 'GBM':
            print('No tuning for Survival GBM because unprone to overfitting and time consuming to train.')
            # Fixed parameters
            self.best_hyperparameters = {'loss': 'coxph', 'criterion': 'friedman_mse', 'random_state': self.seed,
                                         'verbose': 2}
            # Parameters that could be tuned
            self.best_hyperparameters.update({'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3})
        elif self.target == 'Survival' and self.algo_name == 'NeuralNetwork':
            net = tt.practical.MLPVanilla(self.Xs['train'].shape[1], num_nodes=[256, 128, 64, 32], out_features=1,
                                                  batch_norm=True, dropout=0.0, output_bias=False)
            self.model = CoxPH(net, tt.optim.Adam)
            lrfinder = self.model.lr_finder(self.Xs['train'], self.Ys['train'], batch_size=16384, tolerance=10)
            self.model.optimizer.set_lr(lrfinder.get_best_lr())
            self.best_hyperparameters = {}
        elif self.target == 'Survival' and self.algo_name == 'CNN':
            print('No tuning for now. Using the hyperparameters tuned on age prediction.')
            if self.predictors == 'demographics+PWA':
                self.best_hyperparameters = {'CNN_BatchNormalization': True, 'DNN_BatchNormalization': True,
                                             'dropout_rate': 0.7474966212387502, 'learning_rate': 0.0419310826080069,
                                             'weight_decay': 0.15574044327777026}
            elif self.predictors == 'features+PWA':
                self.best_hyperparameters = {'CNN_BatchNormalization': False, 'DNN_BatchNormalization': True,
                                             'dropout_rate': 0.16208346508465238, 'learning_rate': 0.006913480384188409,
                                             'weight_decay': 0.1208198441542345}
            elif self.predictors == 'all+PWA':
                self.best_hyperparameters = {'CNN_BatchNormalization': True, 'DNN_BatchNormalization': True,
                                             'dropout_rate': 0.8392621191523529, 'learning_rate': 0.017722841481230076,
                                             'weight_decay': 0.008323374334617646}
        elif self.target == 'Age' and self.algo_name == 'NeuralNetwork':
            print('No tuning for now.')
            self.best_hyperparameters = {'learning_rate_init': 0.0001, 'alpha': 0.0, 'solver': 'adam',
                                         'activation': 'relu', 'hidden_layer_sizes': (256, 128, 64, 32),
                                         'batch_size': 16384, 'shuffle': True, 'max_iter': 1000,
                                         'early_stopping': True, 'n_iter_no_change': 10, 'verbose': 1}
        #elif self.target == 'Age' and self.algo_name == 'CNN':
        #    print('No tuning for now.')
        #    self.best_hyperparameters = {'learning_rate': 0.001, 'weight_decay': 0.0, 'dropout_rate': 0.05,
        #                                 'CNN_BatchNormalization': True, 'DNN_BatchNormalization': True}
        else:
            trials = Trials()
            self.best_hyperparameters = fmin(fn=partial(self._objective, algo_name=self.algo_name),
                                             space=self.SPACES[self.target][self.algo_name], trials=trials,
                                             algo=tpe.suggest, max_evals=self.n_hyperopt_evals[self.algo_name])
            # Fixing VERY BAD BUG from hyperopt: it returns the *opposite* of the best choice for choice hyperparameters
            print('The best set of hyperparameters selected, before fixing the bug is: ')
            print(self.best_hyperparameters)
            for parameter in ['CNN_BatchNormalization', 'DNN_BatchNormalization']:
                if parameter in self.best_hyperparameters.keys():
                    self.best_hyperparameters[parameter] = not self.best_hyperparameters[parameter]
            # Converting floats to integer (hyperopt did not do it, although the distributions are over integers)
            for parameter in ['num_leaves', 'min_child_samples', 'n_estimators']:
                if parameter in self.best_hyperparameters.keys():
                    self.best_hyperparameters[parameter] = int(self.best_hyperparameters[parameter])
        print('The best set of hyperparameters selected is: ')
        print(self.best_hyperparameters)
    
    def train_model(self):
        if (self.algo_name in ['ElasticNet', 'GBM']) or ((self.target == 'Age') and (self.algo_name == 'NeuralNetwork')):
            self.model = self.algorithms[self.algo_name](**self.best_hyperparameters)
        if self.target == 'Age':
            if self.algo_name == 'CNN':
                self.model = \
                    self._generate_age_CNN(learning_rate=self.best_hyperparameters['learning_rate'],
                                           weight_decay=self.best_hyperparameters['weight_decay'],
                                           dropout_rate=self.best_hyperparameters['dropout_rate'],
                                           CNN_BatchNormalization=self.best_hyperparameters['CNN_BatchNormalization'],
                                           DNN_BatchNormalization=self.best_hyperparameters['DNN_BatchNormalization'])
                '''
                Plot CNN architecture. Does not work on O2
                plot_model(self.model, show_shapes=True,
                           to_file='../figures/CNN_architecture_' + self.target + '_' + self.predictors + '.jpg')
                '''
                model_checkpoint = ModelCheckpoint(self.path_weights, monitor='val_RMSE', mode='min', verbose=1,
                                                   save_best_only=True, save_weights_only=True)
                early_stopping = EarlyStopping(monitor='val_RMSE', min_delta=0, patience=200, verbose=0, mode='min',
                                               restore_best_weights=True)
                reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='min',
                                                         min_delta=0.0)
                self.model.fit(self.Xs['train'], self.Ys['train'], validation_data=(self.Xs['val'], self.Ys['val']),
                               callbacks=[model_checkpoint, early_stopping, reduce_lr_on_plateau], epochs=100000,
                               batch_size=32768, shuffle=True, verbose=0)
                #self.model.load_weights(self.path_weights)
            else:
                self.model.fit(self.Xs['train'], self.Ys['train'])
        elif self.target == 'Survival':
            if self.algo_name == 'ElasticNet':
                df_train = self.Ys['train'].merge(self.Xs['train'], right_index=True, left_index=True)
                self.model.fit(df_train, duration_col='FollowUpTime', event_col='Death')
            elif self.algo_name == 'GBM':
                self.model.fit(self.Xs['train'], self.Ys['train'])
            elif self.algo_name == 'NeuralNetwork':
                callbacks = [tt.callbacks.EarlyStopping(metric='loss', dataset='val', patience=10)]
                self.model.fit(self.Xs['train'], self.Ys['train'], val_data=(self.Xs['val'], self.Ys['val']),
                               batch_size=16384, epochs=512, callbacks=callbacks, verbose=False)
            elif self.algo_name == 'CNN':
                self.model = \
                    self._generate_surv_CNN(weight_decay=self.best_hyperparameters['weight_decay'],
                                            dropout_rate=self.best_hyperparameters['dropout_rate'],
                                            CNN_BatchNormalization=self.best_hyperparameters['CNN_BatchNormalization'],
                                            DNN_BatchNormalization=self.best_hyperparameters['DNN_BatchNormalization'])
                train_fn = InputFunction(self.Xs['train'][0], self.Xs['train'][1], self.Ys['train']['FollowUpTime'],
                                         self.Ys['train']['Death'], drop_last=True, shuffle=True)
                train_predonly_fn = InputFunction(self.Xs['train'][0], self.Xs['train'][1],
                                                  self.Ys['train']['FollowUpTime'], self.Ys['train']['Death'])
                eval_fn = InputFunction(self.Xs['val'][0], self.Xs['val'][1], self.Ys['val']['FollowUpTime'],
                                        self.Ys['val']['Death'])
                test_fn = InputFunction(self.Xs['test'][0], self.Xs['test'][1], self.Ys['test']['FollowUpTime'],
                                        self.Ys['test']['Death'])
                trainer = TrainAndEvaluateModel(model=self.model, num_epochs=200,
                                                learning_rate=self.best_hyperparameters['learning_rate'],
                                                train_dataset=train_fn(), train_predonly_dataset=train_predonly_fn(),
                                                eval_dataset=eval_fn(), test_dataset=test_fn())
                trainer.train_and_evaluate()
                self.trainer = trainer
    
    def evaluate_model_performance(self):
        if self.target == 'Survival' and self.algo_name == 'CNN':
            preds = self.trainer.generate_predictions()
        for fold in self.folds:
            if self.target == 'Age':
                self.PREDs[fold] = self.DATA_FEATUREs[fold][['Age']]
                self.PREDs[fold]['pred'] = self.model.predict(self.Xs[fold])
                self.PERFs[fold] = [r2_score(self.Ys[fold], self.PREDs[fold]['pred'])]
                #print('RMSE'); print(fold); print(np.sqrt(mean_squared_error(self.Ys[fold], self.PREDs[fold]['pred'])))
            elif self.target == 'Survival':
                # Generate predictions
                if self.algo_name == 'ElasticNet':
                    df = self.Ys[fold].merge(self.Xs[fold], right_index=True, left_index=True)
                    self.PREDs[fold]['pred'] = self.model.predict_partial_hazard(df)
                elif self.algo_name == 'CNN':
                    self.PREDs[fold]['pred'] = preds[fold]
                else:
                    self.PREDs[fold]['pred'] = self.model.predict(self.Xs[fold])
                # Compute performances
                self.PERFs[fold] = [concordance_index(self.PREDs[fold]['FollowUpTime'], -self.PREDs[fold]['pred'],
                                                      self.PREDs[fold]['Death'])]
            print(fold + ' fold, score = ' + str(round(self.PERFs[fold][0], 3)))
    
    def generate_feature_importance(self):
        if self.algo_name == 'ElasticNet':
            if self.target == 'Age':
                self.Feature_importance = pd.Series(self.model.coef_)
                self.Feature_importance.name = 'Feature_Importance'
                self.Feature_importance.index = self.Xs['test'].columns.values
            elif self.target == 'Survival':
                self.Feature_importance = self.model.params_
                self.Feature_importance.name = 'Feature_importance'
            order = self.Feature_importance.map(lambda x: x).abs().sort_values(ascending=False)
            self.Feature_importance = self.Feature_importance[order.index]
        elif self.algo_name == 'GBM':
            if self.target == 'Age':
                Feature_importance = pd.Series(self.model.feature_importances_)
                Feature_importance.name = 'Feature_Importance'
                Feature_importance.index = self.Xs['train'].columns.values
                Feature_importance.sort_values(ascending=False, inplace=True)
                Feature_importance /= Feature_importance.sum()
                self.Feature_importance = Feature_importance
            elif self.target == 'Survival':
                Feature_importance = pd.Series(self.model.feature_importances_)
                Feature_importance.name = 'Feature_Importance'
                Feature_importance.index = self.Xs['train'].columns.values
                Feature_importance.sort_values(ascending=False, inplace=True)
                self.Feature_importance = Feature_importance
        if self.Feature_importance is not None:
            print('Feature importance:')
            print(self.Feature_importance)
    
    def save_predictions(self):
        # Save performances
        Perfs = pd.DataFrame.from_dict(self.PERFs)
        Perfs.to_csv('../data/Performances_' + self.target + '_' + self.predictors + '_' + self.algo_name + '.csv',
                     index=False)
        # Save predictions
        for fold in self.folds:
            self.PREDs[fold].to_csv('../data/Predictions_' + self.target + '_' + self.predictors + '_' +
                                    self.algo_name + '_' + fold + '.csv', index=False)
        
        # Save feature importance
        if self.Feature_importance is not None:
            self.Feature_importance.to_csv('../data/Feature_importance_' + self.target + '_' + self.predictors + '_' +
                                           self.algo_name + '.csv', index=True)


class Postprocessing(Basics):
    
    def __init__(self):
        # Parameters
        Basics.__init__(self)
        self.targets = ['Age', 'Survival']
        self.algorithms = ['ElasticNet', 'GBM', 'NeuralNetwork', 'CNN']
        self.predictors_scalars = ['demographics', 'features', 'PWA', 'all']
        self.predictors_CNN = ['demographics+PWA', 'features+PWA', 'all+PWA']
        self.predictors_all = self.predictors_scalars + self.predictors_CNN
        self.PERFORMANCES = {}
        self.n_bootstrap_iterations = {'Age': 1000, 'Survival': 100}
        
        # Initiate empty dataframe
        Performances = np.empty((len(self.algorithms), len(self.predictors_all),))
        Performances.fill(np.nan)
        Performances = pd.DataFrame(Performances)
        Performances.index = self.algorithms
        Performances.columns = self.predictors_all
        
        for target in self.targets:
            self.PERFORMANCES[target] = {}
            for fold in self.folds:
                self.PERFORMANCES[target][fold] = Performances.copy()

    def _bootstrap(self, data, target):
        # Compute main score
        if target == 'Age':
            main = r2_score(data['Age'], data['pred'])
        elif target == 'Survival':
            main = concordance_index(data['FollowUpTime'], -data['pred'], data['Death'])
        # Compute std
        scores = []
        for i in range(self.n_bootstrap_iterations[target]):
            data_i = resample(data, replace=True, n_samples=len(data.index))
            if target == 'Age':
                score_i = r2_score(data_i['Age'], data_i['pred'])
            elif target == 'Survival':
                score_i = concordance_index(data_i['FollowUpTime'], -data_i['pred'], data_i['Death'])
            scores.append(score_i)
            std = np.std(scores)
        return str(round(main * 100, 1)) + '+-' + str(round(std * 100, 1))
    
    def compute_performances(self):
        for target in self.targets:
            for algorithm in self.algorithms:
                if algorithm == 'CNN':
                    predictorsS = self.predictors_CNN
                else:
                    predictorsS = self.predictors_scalars
                for predictors in predictorsS:
                    for fold in self.folds:
                        path_preds = '../data/Predictions_' + target + '_' + predictors + '_' + algorithm + '_' + \
                                     fold + '.csv'
                        if os.path.exists(path_preds):
                            Predictions = pd.read_csv(path_preds)
                            if target == 'Age':
                                score = self._bootstrap(Predictions, target)
                            elif target == 'Survival':
                                score = self._bootstrap(Predictions, target)
                            self.PERFORMANCES[target][fold].loc[algorithm, predictors] = score
            # Print performances
            print('Performance for ' + target + ': ')
            print(self.PERFORMANCES[target])
            
    def generate_barplots(self):
        # Age
        Perf = self.PERFORMANCES['Age']['test'].astype(str).applymap(lambda x: float(x.split('+')[0]))
        fig = Perf.transpose().plot.bar(ylabel='R-Squared (%)', title='R-Squared = f(Dataset, Algorithm)')
        fig.figure.savefig('../figures/R2s.png')
        
        # Survival
        Perf = self.PERFORMANCES['Survival']['test'].astype(str).applymap(lambda x: float(x.split('+')[0]))
        fig = Perf.transpose().plot.bar(ylabel='C-Index (%)', title='Concordance Index = f(Dataset, Algorithm)')
        fig.figure.savefig('../figures/CIs.png')
        
        '''
        Code to generate regression plot
        Perf = pd.read_csv("../data/Performances_Age_test.csv", index_col=0)
        Perf= Perf.astype(str).applymap(lambda x: float(x.split('+')[0]))
        fig = Perf.transpose().plot.bar(ylabel='R-Squared (%)', title='R-Squared = f(Dataset, Algorithm)')
        fig.figure.savefig('../figures/R2s.png')
        '''
    
    def save_data(self):
        for target in self.targets:
            for fold in self.folds:
                self.PERFORMANCES[target][fold].to_csv('../data/Performances_' + target + '_' + fold + '.csv')
