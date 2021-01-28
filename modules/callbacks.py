# -*- coding: utf-8 -*-

from fastNLP import Callback, Tester, DataSet


class EvaluateCallback(Callback):

    def __init__(self, data=None, tester=None):

        super().__init__()
        self.datasets = {}
        self.testers = {}
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(
                            f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(
                    value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError(
                "Trainer has no dev data, you cannot pass extra DataSet to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=1,  # self.trainer.kwargs.get(
                                # 'dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics, verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):

        if len(self.testers) > 0:
            for idx, (key, tester) in enumerate(self.testers.items()):
                try:
                    eval_result = tester.test()
                    if idx == 0:
                        indicator, indicator_val = _check_eval_results(
                            eval_result)
                        if indicator_val > self.best_test_metric_sofar:
                            self.best_test_metric_sofar = indicator_val
                            self.best_test_epoch = self.epoch
                            self.best_test_sofar = eval_result
                    if better_result:
                        self.best_dev_test = eval_result
                        self.best_dev_epoch = self.epoch
                    self.logger.info(
                        "EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))
                except Exception as e:
                    self.logger.error(
                        "Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        if self.best_test_sofar:
            self.logger.info("Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_test_sofar, self.best_test_epoch))
        if self.best_dev_test:
            self.logger.info("Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_dev_test, self.best_dev_epoch))


def _check_eval_results(metrics, metric_key=None):
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[
                0], list(metric_dict.keys())[0]
        else:
            if metric_key not in metric_dict:
                raise RuntimeError(
                    f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format(
            (tuple, dict), type(metrics)))
    return indicator, indicator_val
