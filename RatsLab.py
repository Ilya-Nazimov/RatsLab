import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pyedflib
import pandas as pd
from catboost import CatBoostClassifier
import csv


def calculate_statistics_per_second(edf_file_path):
    # Открываем EDF файл
    with pyedflib.EdfReader(edf_file_path) as f:
        n_channels = f.signals_in_file
        sampling_freq = int(f.getSampleFrequency(0))
        total_samples = f.getNSamples()[0]  # Получаем общее количество выборок для первого канала

        # Рассчитываем количество секунд в записи и приводим к целому числу
        total_seconds = int(total_samples // sampling_freq)

        # Инициализируем массивы для хранения средних значений и стандартных отклонений
        means = np.zeros((total_seconds, n_channels))
        std_devs = np.zeros((total_seconds, n_channels))
        avg_freqs = np.zeros((total_seconds, n_channels))
        std_freqs = np.zeros((total_seconds, n_channels))

        for i in range(n_channels):
            channel_data = f.readSignal(i)

            # Обрабатываем данные по секундам
            for sec in range(total_seconds):
                start_index = sec * sampling_freq
                end_index = start_index + sampling_freq
                second_data = channel_data[start_index:end_index]

                # Вычисляем среднее значение и стандартное отклонение
                means[sec, i] = np.mean(second_data)
                std_devs[sec, i] = np.std(second_data)

                # Выполняем быстрое преобразование Фурье
                fft_result = np.fft.fft(second_data)
                fft_magnitude = np.abs(fft_result)[:sampling_freq // 2]  # Берем только положительные частоты
                frequencies = np.fft.fftfreq(sampling_freq, d=1 / sampling_freq)[:sampling_freq // 2]  # Частоты

                # Находим среднюю частоту (взвешенную по мощности)
                power_spectrum = fft_magnitude ** 2
                total_power = np.sum(power_spectrum)
                if total_power > 0:
                    avg_freqs[sec, i] = np.sum(frequencies * power_spectrum) / total_power
                    std_freqs[sec, i] = np.sqrt(
                        np.sum(((frequencies - avg_freqs[sec, i]) ** 2) * power_spectrum) / total_power)
                else:
                    avg_freqs[sec, i] = 0
                    std_freqs[sec, i] = 0

        # Создаем массивы для сдвинутых значений
        means_up = np.roll(means, -1, axis=0)
        means_down = np.roll(means, 1, axis=0)

        std_devs_up = np.roll(std_devs, -1, axis=0)
        std_devs_down = np.roll(std_devs, 1, axis=0)

        avg_freqs_up = np.roll(avg_freqs, -1, axis=0)
        avg_freqs_down = np.roll(avg_freqs, 1, axis=0)

        std_freqs_up = np.roll(std_freqs, -1, axis=0)
        std_freqs_down = np.roll(std_freqs, 1, axis=0)

        # Удаляем крайние значения (если необходимо)
        means_up[-1] = means_up[-2]  # Заменяем последний элемент на предпоследний
        means_down[0] = means_down[1]  # Заменяем первый элемент на второй

        std_devs_up[-1] = std_devs_up[-2]
        std_devs_down[0] = std_devs_down[1]

        avg_freqs_up[-1] = avg_freqs_up[-2]
        avg_freqs_down[0] = avg_freqs_down[1]

        std_freqs_up[-1] = std_freqs_up[-2]
        std_freqs_down[0] = std_freqs_down[1]

    return (means, std_devs, avg_freqs, std_freqs,
            means_up, std_devs_up, avg_freqs_up, std_freqs_up,
            means_down, std_devs_down, avg_freqs_down, std_freqs_down)


def create_annotations(labels):
    annotations = []
    n = len(labels)

    for i in range(n):
        current_label = labels[i]

        # Если это первая метка или она отличается от предыдущей
        if i == 0 or current_label != labels[i - 1]:
            # Записываем начало текущей метки
            annotations.append(["+" + str(i + 1), str(current_label + "1")[2:-2]])  # Начало

        # Если это последняя метка или следующая метка отличается
        if i == n - 1 or current_label != labels[i + 1]:
            # Записываем конец текущей метки
            annotations.append(["+" + str(i + 1), str(current_label + "2")[2:-2]])  # Конец

    filename = 'annotations.csv'

    # Запись данных в CSV файл
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Запись заголовков
        writer.writerow(['Onset', 'Annotation'])

        # Запись данных
        for row in annotations:
            writer.writerow(row)


class EDFViewer:
    def __init__(self, filename):
        means, std_devs, avg_freqs, std_freqs, means_up, std_devs_up, avg_freqs_up, std_freqs_up, means_down, std_devs_down, avg_freqs_down, std_freqs_down = calculate_statistics_per_second(filename)

        # Создаем DataFrame для хранения результатов
        means_df = pd.DataFrame(means, columns=[f'mean_{i + 1}' for i in range(means.shape[1])])
        std_devs_df = pd.DataFrame(std_devs, columns=[f'meanstd_{i + 1}' for i in range(std_devs.shape[1])])
        avg_freqs_df = pd.DataFrame(avg_freqs, columns=[f'avg_freqs_{i + 1}' for i in range(means.shape[1])])
        std_freqs_df = pd.DataFrame(std_freqs, columns=[f'std_freqs_{i + 1}' for i in range(std_devs.shape[1])])

        means_df_up = pd.DataFrame(means, columns=[f'mean_{i + 1}_up' for i in range(means.shape[1])])
        std_devs_df_up = pd.DataFrame(std_devs, columns=[f'meanstd_{i + 1}_up' for i in range(std_devs.shape[1])])
        avg_freqs_df_up = pd.DataFrame(avg_freqs, columns=[f'avg_freqs_{i + 1}_up' for i in range(means.shape[1])])
        std_freqs_df_up = pd.DataFrame(std_freqs, columns=[f'std_freqs_{i + 1}_up' for i in range(std_devs.shape[1])])

        means_df_down = pd.DataFrame(means, columns=[f'mean_{i + 1}_down' for i in range(means.shape[1])])
        std_devs_df_down = pd.DataFrame(std_devs, columns=[f'meanstd_{i + 1}_down' for i in range(std_devs.shape[1])])
        avg_freqs_df_down = pd.DataFrame(avg_freqs, columns=[f'avg_freqs_{i + 1}_down' for i in range(means.shape[1])])
        std_freqs_df_down = pd.DataFrame(std_freqs,
                                         columns=[f'std_freqs_{i + 1}_down' for i in range(std_devs.shape[1])])

        # Объединяем средние значения и стандартные отклонения
        self.final_results = pd.concat([means_df, std_devs_df, means_df_up, std_devs_df_up, means_df_down, std_devs_df_down,
                                   avg_freqs_df, std_freqs_df, avg_freqs_df_up, std_freqs_df_up, avg_freqs_df_down,
                                   std_freqs_df_down], axis=1)

        self.X = self.final_results[['mean_1', 'mean_2', 'mean_3', 'meanstd_1', 'meanstd_2', 'meanstd_3',
                      'mean_1_up','mean_2_up','mean_3_up','meanstd_1_up','meanstd_2_up','meanstd_3_up',
                      'mean_1_down','mean_2_down','mean_3_down','meanstd_1_down','meanstd_2_down','meanstd_3_down',
                      "avg_freqs_1","avg_freqs_2","avg_freqs_3","std_freqs_1","std_freqs_2","std_freqs_3",
                      "avg_freqs_1_up","avg_freqs_2_up","avg_freqs_3_up","std_freqs_1_up","std_freqs_2_up","std_freqs_3_up",
                      "avg_freqs_1_down","avg_freqs_2_down","avg_freqs_3_down","std_freqs_1_down","std_freqs_2_down","std_freqs_3_down"]]

        self.model = CatBoostClassifier()
        self.model.load_model('RatLabs_Model')
        self.y_pred = self.model.predict(self.X)
        self.y_pred_proba = self.model.predict_proba(self.X)  # Получаем вероятности предсказаний
        self.results = pd.DataFrame(self.y_pred, columns=['Pred'])
        self.results.to_csv("test_preds", index_label='Sec')

        annotations = create_annotations(self.y_pred)

        self.filename = filename
        self.f = pyedflib.EdfReader(self.filename)
        self.n_channels = self.f.signals_in_file
        self.channels = self.f.getSignalLabels()
        self.data = [self.f.readSignal(i) for i in range(self.n_channels)]
        self.fs = [self.f.getSampleFrequency(i) for i in range(self.n_channels)]
        self.total_time = self.data[0].size / self.fs[0]
        self.time = np.arange(self.data[0].size) / self.fs[0]

        self.width_in_seconds = 10
        self.x_start = 0
        self.x_end = self.width_in_seconds

        # Создаем фигуру и оси для каждого канала
        self.fig, axs = plt.subplots(self.n_channels + 2, 1, figsize=(10, 10))  # Увеличиваем количество подграфиков

        self.lines = []
        for i in range(self.n_channels):
            line, = axs[i].plot(self.time, self.data[i])
            axs[i].set_title(f'Channel: {self.channels[i]}')
            axs[i].set_ylabel('Amplitude')
            axs[i].set_xlim(self.x_start, self.x_end)
            axs[i].grid()
            self.lines.append(line)

        axs[-2].set_xlabel('Time (s)')

        # График предсказаний
        axs[-2].set_title('Predictions')
        axs[-2].set_ylim(-1, 4)
        axs[-2].set_yticks([0, 1, 2, 3])
        axs[-2].set_yticklabels(['ds', 'is', 'awake', 'swd'])

        # Преобразуем предсказания в числовые значения
        prediction_indices = {'ds': 0, 'is': 1, 'awake': 2, 'swd': 3}
        predictions_numeric = [prediction_indices.get(label, -1) for label in self.y_pred.flatten()]
        assert all(pred >= 0 for pred in predictions_numeric), "Некорректные предсказания: есть значения -1"

        # Инициализация линии предсказаний
        self.prediction_line, = axs[-2].step([], [], where='post', color='orange')

        # График уверенности модели
        axs[-1].set_title('Model Confidence')
        axs[-1].set_ylim(0, 1)  # Уверенность нормализована от 0 до 1
        self.confidence_line, = axs[-1].plot([], [], color='blue')
        axs[-1].set_xlabel('Time (s)')

        # Создаем ползунок для навигации
        self.ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
        self.slider = Slider(self.ax_slider, 'Time', 0, self.total_time - self.width_in_seconds, valinit=self.x_start)

        def update(val):
            self.x_start = val
            self.x_end = self.x_start + self.width_in_seconds

            for i in range(self.n_channels):
                # Обновляем данные графиков
                self.lines[i].set_xdata(self.time)
                self.lines[i].set_ydata(self.data[i])
                axs[i].set_xlim(self.x_start, self.x_end)

            # Обновление графика предсказаний
            current_sec_index_start = int(self.x_start * len(predictions_numeric) / self.total_time)
            current_sec_index_end = int(self.x_end * len(predictions_numeric) / self.total_time)

            if current_sec_index_end <= len(predictions_numeric):
                prediction_segment = predictions_numeric[current_sec_index_start:current_sec_index_end]
                time_segment = np.linspace(self.x_start, self.x_end, num=len(prediction_segment))
                # Обновление линии предсказаний
                if len(prediction_segment) > 0:
                    self.prediction_line.set_xdata(time_segment)
                    self.prediction_line.set_ydata(prediction_segment)
                    axs[-2].set_xlim(self.x_start, self.x_end)

                # Обновление графика уверенности
                confidence_segment = np.max(self.y_pred_proba[current_sec_index_start:current_sec_index_end], axis=1)
                confidence_time_segment = np.linspace(self.x_start, self.x_end, num=len(confidence_segment))

                if len(confidence_segment) > 0:
                    self.confidence_line.set_xdata(confidence_time_segment)
                    self.confidence_line.set_ydata(confidence_segment)
                    axs[-1].set_xlim(self.x_start, self.x_end)

            plt.draw()  # Перерисовываем график

        # Привязываем обновление к ползунку
        self.slider.on_changed(update)

    def plot(self):
        plt.show()

if __name__ == "__main__":
    filename = 'TEST2_30min.edf'  # Замените на ваш файл EDF
    viewer = EDFViewer(filename)
    viewer.plot()
