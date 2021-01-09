#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats
import re
import os

def detect_peaks(ecg_measurements, signal_frequency, gain):

        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.

        This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. 
        It was created and used for experimental purposes in psychophysiology and psychology.
        You can find more information in module documentation:
        https://github.com/c-labpl/qrs_detector
        If you use these modules in a research project, please consider citing it:
        https://zenodo.org/record/583770
        If you use these modules in any other project, please refer to MIT open-source license.

        If you have any question on the implementation, please refer to:

        Michal Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
        Marta lukowska (Jagiellonian University)
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/c-labpl/qrs_detector
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        
        MIT License
        Copyright (c) 2017 Michal Sznajder, Marta Lukowska
    
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        """

        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1

        # Change proportionally when adjusting frequency (in samples).
        integration_window = 30
        findpeaks_limit = 0.35
        # Change proportionally when adjusting frequency (in samples).
        findpeaks_spacing = 100
        # Change proportionally when adjusting frequency (in samples).
        # refractory_period = 240
        # qrs_peak_filtering_factor = 0.125
        # noise_peak_filtering_factor = 0.125
        # qrs_noise_diff_weight = 0.25

        # Detection results.
        # qrs_peaks_indices = np.array([], dtype=int)
        # noise_peaks_indices = np.array([], dtype=int)

        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(
            ecg_measurements,
            lowcut=filter_lowcut,
            highcut=filter_highcut,
            signal_freq=signal_frequency,
            filter_order=filter_order
        )

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(
            squared_ecg_measurements,
            np.ones(integration_window) / integration_window
        )

        # Fiducial mark - peak detection on integrated measurements.
        detected_peaks_indices = findpeaks(
            data=integrated_ecg_measurements,
            limit=findpeaks_limit,
            spacing=findpeaks_spacing
        )

        detected_peaks_values = integrated_ecg_measurements[
            detected_peaks_indices
        ]

        return detected_peaks_values, detected_peaks_indices


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def findpeaks(data, spacing=1, limit=None):
    """
    Janko Slavic peak detection algorithm and implementation.
    https://github.com/jankoslavic/py-tools/tree/master/findpeaks
    Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param ndarray data: data
    :param float spacing: minimum spacing to the next peak
        (should be 1 or more)
    :param float limit: peaks should have value greater or equal
    :return array: detected peaks indexes array
    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(
            peak_candidate,
            np.logical_and(h_c > h_b, h_c > h_a)
        )

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def lead_info(lines):
    leads = []
    for line in lines:
        _, _, gain, _, _, _, _, _, lead_name = line.split()
        gain = float(gain.split("/")[0])
        leads.append((lead_name, gain))
    return leads


def get_metadata_from_file(header_filepath):
    with open(header_filepath, 'r') as f:
        header = f.read()
    
    metadata = get_metadata_from_header(header)

    return metadata


def get_metadata_from_header(header):

    def process(string_list):
        string_list = [x.replace(" ", "") for x in string_list]
        if len(string_list) == 1 and string_list[0] == "Unknown":
            return []
        return string_list

    def get_first_item(_list, split=True):
        if len(_list) > 0:
            first_item = _list[0]
            if split:
                first_item = first_item.split(",")
            return first_item
        return []

    first_line = header.split("\n")
    first_line = first_line[0]
    subject_id, n_leads, sr, _, date, hour = first_line.split()
    age = get_first_item(re.findall("#Age:\s*([0-9a-zA-Z, -]*)", header), split=False)
    sex = get_first_item(re.findall("#Sex:\s*([0-9a-zA-Z, ]*)", header), split=False)
    dx = get_first_item(re.findall("#Dx:\s*([0-9a-zA-Z, ]*)", header))
    rx = get_first_item(re.findall("#Rx:\s*([0-9a-zA-Z, ]*)", header))
    hx = get_first_item(re.findall("#Hx:\s*([0-9a-zA-Z, ]*)", header))
    sx = get_first_item(re.findall("#Sx:\s*([0-9a-zA-Z, ]*)", header))

    subject_id = os.path.splitext(subject_id)[0]

    n_leads = int(n_leads)
    name, gain = zip(*lead_info(header.split("\n")[1:1+n_leads]))
    result = {
        "subject_id": subject_id,
        "leads_names": name,
        "leads_gains": gain,
        "n_leads": int(n_leads),
        "sr": float(sr),
        "date": date,
        "hour": hour,
        "age": 57 if age in ("NaN", "-1") else int(age),
        "sex": int(sex == "Female"),
        "dx": process(dx),
        "rx": process(rx),
        "hx": process(hx),
        "sx": process(sx),
    }
    return result


def baseline_features(data, metadata):

    assert len(data) == metadata["n_leads"] == 12
    sr = metadata["sr"]
    features = [metadata["age"], metadata["sex"]]

    for i in range(len(data)):
        gain = metadata["leads_gains"][i]

        peaks, idx = detect_peaks(data[i], sr, gain)

    #   mean
        mean_RR = np.mean(idx / sr * 1000)
        mean_Peaks = np.mean(peaks * gain)

    #   median
        median_RR = np.median(idx / sr * 1000)
        median_Peaks = np.median(peaks * gain)

    #   standard deviation
        std_RR = np.std(idx / sr * 1000)
        std_Peaks = np.std(peaks * gain)

    #   variance
        var_RR = stats.tvar(idx / sr * 1000)
        var_Peaks = stats.tvar(peaks * gain)

    #   Skewness
        skew_RR = stats.skew(idx / sr * 1000)
        skew_Peaks = stats.skew(peaks * gain)

    #   Kurtosis
        kurt_RR = stats.kurtosis(idx / sr * 1000)
        kurt_Peaks = stats.kurtosis(peaks * gain)

        features.extend(
            [
                mean_RR, mean_Peaks, median_RR, median_Peaks,
                std_RR, std_Peaks, var_RR, var_Peaks, skew_RR,
                skew_Peaks, kurt_RR, kurt_Peaks
            ]
        )

    features = np.hstack(features)

    return features
