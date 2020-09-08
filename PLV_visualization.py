import mne
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from math import floor
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
import math
import pandas as pd
import csv


seizures = {
    'chb01_03.edf': [(2996, 3036)],
    'chb01_04.edf': [(1467, 1494)],
    'chb01_15.edf': [(1732, 1772)],
    'chb01_16.edf': [(1015, 1066)],
    'chb01_18.edf': [(1720, 1810)],
    'chb01_21.edf': [(327, 420)],
    'chb01_26.edf': [(1862, 1963)],
    'chb02_16.edf': [(130, 212)],
    'chb02_16+.edf': [(2972, 3053)],
    'chb02_19.edf': [(3369, 3378)],
    'chb03_01.edf': [(362, 414)],
    'chb03_02.edf': [(731, 796)],
    'chb03_03.edf': [(432, 501)],
    'chb03_04.edf': [(2162, 2214)],
    'chb03_34.edf': [(1982, 2029)],
    'chb03_35.edf': [(2592, 2656)],
    'chb03_36.edf': [(1725, 1778)],
    'chb05_06.edf': [(417, 532)],
    'chb05_13.edf': [(1086, 1196)],
    'chb05_16.edf': [(2317, 2413)],
    'chb05_17.edf': [(2451, 2571)],
    'chb05_22.edf': [(2348, 2465)],
    'chb06_01.edf': [(1724, 1738), (7461, 7476), (13525, 13540)],
    'chb06_04.edf': [(327, 347), (6211, 6231)],
    'chb06_09.edf': [(12500, 12516)],
    'chb06_10.edf': [(10833, 10845)],
    'chb06_13.edf': [(506, 519)],
    'chb06_18.edf': [(7799, 7811)],
    'chb06_24.edf': [(9387, 9403)],
    'chb07_12.edf': [(4920, 5006)],
    'chb07_13.edf': [(3285, 3381)],
    'chb07_19.edf': [(13688, 13831)],
    'chb08_02.edf': [(2670, 2841)],
    'chb08_05.edf': [(2856, 3046)],
    'chb08_11.edf': [(2988, 3122)],
    'chb08_13.edf': [(2417, 2577)],
    'chb08_21.edf': [(2083, 2347)],
    'chb09_06.edf': [(12231, 12295)],
    'chb09_08.edf': [(2951, 3030), (9196, 9267)],
    'chb09_19.edf': [(5299, 5361)],
    'chb10_12.edf': [(6313, 6348)],
    'chb10_20.edf': [(6888, 6958)],
    'chb10_27.edf': [(2382, 2447)],
    'chb10_30.edf': [(3021, 3079)],
    'chb10_31.edf': [(3801, 3877)],
    'chb10_38.edf': [(4618, 4707)],
    'chb10_89.edf': [(1383, 1437)],
    'chb11_82.edf': [(298, 320)],
    'chb11_92.edf': [(2695, 2727)],
    'chb11_99.edf': [(1454, 2206)],
    'chb12_06.edf': [(1665, 1726), (3415, 3447)],
    'chb12_08.edf': [(1426, 1439), (1591, 1614), (1957, 1977), (2798, 2824)],
    'chb12_10.edf': [(593, 625), (811, 856)],
    'chb12_11.edf': [(1085, 1122)],
    'chb12_23.edf': [(253, 333), (425, 522), (630, 670)],
    'chb12_27.edf': [(916, 951), (1097, 1124), (1728, 1753), (1921, 1963), (2388, 2440), (2621, 2669)],
    'chb12_28.edf': [(181, 215)],
    'chb12_29.edf': [(107, 146), (554, 592), (1163, 1199), (1401, 1447), (1884, 1921), (3557, 3584)],
    'chb12_33.edf': [(2185, 2206), (2427, 2450)],
    'chb12_36.edf': [(653, 680)],
    'chb12_38.edf': [(1548, 1573), (2798, 2821), (2966, 3009), (3146, 3201), (3364, 3410)],
    'chb12_42.edf': [(699, 750), (945, 973), (1170, 1199), (1676, 1701), (2213, 2236)],
    'chb13_19.edf': [(2077, 2121)],
    'chb13_21.edf': [(934, 1004)],
    'chb13_40.edf': [(142, 173), (530, 594)],
    'chb13_55.edf': [(458, 478), (2436, 2454)],
    'chb13_58.edf': [(2474, 2491)],
    'chb13_59.edf': [(3339, 3401)],
    'chb13_60.edf': [(638, 660)],
    'chb13_62.edf': [(851, 916), (1626, 1691), (2664, 2721)],
    'chb14_03.edf': [(1986, 2000)],
    'chb14_04.edf': [(1372, 1392), (2817, 2839)],
    'chb14_06.edf': [(1911, 1925)],
    'chb14_11.edf': [(1838, 1879)],
    'chb14_17.edf': [(3239, 3259)],
    'chb14_18.edf': [(1039, 1061)],
    'chb14_27.edf': [(2833, 2849)],
    'chb15_06.edf': [(272, 397)],
    'chb15_10.edf': [(1082, 1113)],
    'chb15_15.edf': [(1591, 1748)],
    'chb15_17.edf': [(1925, 1960)],
    'chb15_20.edf': [(607, 662)],
    'chb15_22.edf': [(760, 965)],
    'chb15_28.edf': [(876, 1066)],
    'chb15_31.edf': [(1751, 1871)],
    'chb15_40.edf': [(834, 894), (2378, 2497), (3362, 3425)],
    'chb15_46.edf': [(3322, 3429)],
    'chb15_49.edf': [(1108, 1248)],
    'chb15_52.edf': [(778, 849)],
    'chb15_54.edf': [(263, 318), (843, 1020), (1524, 1595), (2179, 2250), (3428, 3460)],
    'chb15_62.edf': [(751, 859)],
    'chb16_10.edf': [(2290, 2299)],
    'chb16_11.edf': [(1120, 1129)],
    'chb16_14.edf': [(1854, 1868)],
    'chb16_16.edf': [(1214, 1220)],
    'chb16_17.edf': [(227, 236), (1694, 1700), (2162, 2170), (3290, 3298)],
    'chb16_18.edf': [(627, 635)],
    'chb17a_03.edf': [(2282, 2372)],
    'chb17a_04.edf': [(3025, 3140)],
    'chb17b_63.edf': [(3136, 3224)],
    'chb18_29.edf': [(3477, 3527)],
    'chb18_30.edf': [(541, 571)],
    'chb18_31.edf': [(2087, 2155)],
    'chb18_32.edf': [(1908, 1963)],
    'chb18_35.edf': [(2196, 2264)],
    'chb18_36.edf': [(463, 509)],
    'chb19_28.edf': [(299, 377)],
    'chb19_29.edf': [(2964, 3041)],
    'chb19_30.edf': [(3159, 3240)],
    'chb20_12.edf': [(94, 123)],
    'chb20_13.edf': [(1440, 1470), (2498, 2537)],
    'chb20_14.edf': [(1971, 2009)],
    'chb20_15.edf': [(390, 425), (1689, 1738)],
    'chb20_16.edf': [(2226, 2261)],
    'chb20_68.edf': [(1393, 1432)],
    'chb21_19.edf': [(1288, 1344)],
    'chb21_20.edf': [(2627, 2677)],
    'chb21_21.edf': [(2003, 2084)],
    'chb21_22.edf': [(2553, 2565)],
    'chb22_20.edf': [(3367, 3425)],
    'chb22_25.edf': [(3139, 3213)],
    'chb22_38.edf': [(1263, 1335)],
    'chb23_06.edf': [(3962, 4075)],
    'chb23_08.edf': [(325, 345), (5104, 5151)],
    'chb23_09.edf': [(2589, 2660), (6885, 6947), (8505, 8532), (9580, 9664)],
    'chb24_01.edf': [(480, 505), (2451, 2476)],
    'chb24_03.edf': [(231, 260), (2883, 2908)],
    'chb24_04.edf': [(1088, 1120), (1411, 1438), (1745, 1764)],
    'chb24_06.edf': [(1229, 1253)],
    'chb24_07.edf': [(38, 60)],
    'chb24_09.edf': [(1745, 1764)],
    'chb24_11.edf': [(3527, 3597)],
    'chb24_13.edf': [(3288, 3304)],
    'chb24_14.edf': [(1939, 1966)],
    'chb24_15.edf': [(3552, 3569)],
    'chb24_17.edf': [(3515, 3581)],
    'chb24_21.edf': [(2804, 2872)],
}


def patient_channels(file):
    if 'chb01' in file: return ['FP1-F7', 'FP1-F3', 'FP2-F4', 'FP2-F8', 'FT9-FT10']
    if 'chb02' in file: return ['FP1-F7', 'FP1-F3', 'FP2-F4', 'FP2-F8', 'FT9-FT10']
    if 'chb03' in file: return ['C3-P3', 'P8-O2', 'CZ-PZ', 'T7-P7', 'P3-O1']
    if 'chb08' in file: return ['C3-P3', 'P7-O1', 'T7-P7', 'C4-P4', 'CZ-PZ']
    if 'chb14' in file: return ['C3-P3', 'C4-P4', 'F3-C3', 'F4-C4', 'FZ-CZ']
    if 'chb15' in file: return ['P7-T7', 'P3-O1', 'C3-P3', 'T7-P7', 'P7-O1']
    if 'chb16' in file: return ['F4-C4', 'P4-O2', 'C3-P3', 'FZ-CZ', 'CZ-PZ']
    if 'chb17' in file: return ['P3-O1', 'C3-P3', 'P4-O2', 'CZ-PZ', 'P7-T7']
    if 'chb18' in file: return ['P4-O2', 'T8-P8-0', 'FT10-T8', 'T8-P8-1', 'P8-O2']
    if 'chb19' in file: return ['F3-C3', 'FZ-CZ', 'CZ-PZ', 'C4-P4', 'C3-P3']
    if 'chb20' in file: return ['T7-P7', 'C3-P3', 'F8-T8', 'F4-C4', 'T7-FT9']
    if 'chb21' in file: return ['CZ-PZ', 'C3-P3', 'P7-O1', 'P3-O1', 'F7-T7']
    if 'chb23' in file: return ['C3-P3', 'FZ-CZ', 'P3-O1', 'P4-O2', 'F3-C3']
    if 'chb24' in file: return ['C3-P3', 'C4-P4', 'P4-O2', 'F4-C4', 'FZ-CZ']
    else: return ['FP1-F7', 'FP1-F3', 'FP2-F4', 'FP2-F8', 'FT9-FT10']


def hullMA(data, w=30):
    res = np.zeros(w-1)
    for i in range(w, data.shape[0]+1):
        temp = data[i-w:i]
        whole_average = sum(temp) / w
        half_average = sum(temp[int(w/2):]) / int(w/2)
        res = np.append(res, 2*half_average-whole_average)
    print(res.shape)
    sqp = int(math.sqrt(w))
    fres = np.zeros(sqp-1)
    for i in range(sqp, res.shape[0]+1):
        temp = res[i-sqp:i]
        ma = sum(temp) / sqp
        fres = np.append(fres, ma)
    print(fres.shape)
    return fres

def MA(data, sqp=30):
    fres = np.zeros(sqp - 1)
    for i in range(sqp, data.shape[0] + 1):
        temp = data[i - sqp:i]
        ma = sum(temp) / sqp
        fres = np.append(fres, ma)
    print(fres.shape)
    return fres

def ema(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def plot_plv_vectorized():
    data.pick_channels(['P4-O2', 'C4-P4'])
    recording_length = data.n_times / 256
    if (recording_length % 3) != 0:
        # raw.crop(0, floor(recording_length / t) * t)
        recording_length = floor(recording_length / 3) * 3
    recording_length = int(recording_length)
    data.filter(18, 20)
    sig1 = data.get_data()[0]
    sig2 = data.get_data()[1]
    if sig1.shape != (recording_length * 256):
        sig1 = sig1[0:recording_length * 256]
        sig2 = sig2[0:recording_length * 256]
    sig1 = sig1.reshape((int(recording_length / 3), -1))
    sig2 = sig2.reshape((int(recording_length / 3), -1))
    sig1 = signal.hilbert(sig1)
    sig2 = signal.hilbert(sig2)
    sig1 = np.angle(sig1)
    sig2 = np.angle(sig2)
    sig1 = np.unwrap(sig1)
    sig2 = np.unwrap(sig2)
    complex_phase_diff = np.exp(np.complex(0, 1) * (sig1 - sig2))
    # pld = np.abs(np.sum(complex_phase_diff, axis=-1, keepdims=True))
    # pld = normalize(pld, axis=0)
    p1 = np.sum(np.real(complex_phase_diff), axis=-1)**2
    p2 = np.sum(np.imag(complex_phase_diff), axis=-1)**2
    pld = (np.sqrt(p1+p2)) / (3*256)
    hull = hullMA(pld, 20)
    ma = MA(pld, 20)
    plv = savgol_filter(pld, 51, 3)
    emas = ema(pld, 20)

    # List to hold x values.
    x_number_values = [i for i in range(len(pld))]
    hull_values = [i for i in range(len(hull))]

    # Plot the number in the list and set the line thickness.
    #plt.plot(x_number_values, pld, linewidth=1)
    plt.plot(x_number_values, plv, linewidth=2)
    # plt.plot(hull_values, hull, linewidth=2)
    #plt.plot(x_number_values, ma, linewidth=2)
    # plt.plot(x_number_values, emas, linewidth=2)

    # Set the line chart title and the text font size.
    plt.title("Square Numbers", fontsize=19)
    plt.show()


def plot_stft(data):
    buf = data[0]
    plt.figure(figsize=(30, 20))
    fs = 256

    f, t, Zxx = signal.stft(buf, fs, nperseg=100)
    maximum = 0
    Zxx = np.abs(Zxx)
    Zxx = normalize(Zxx, axis=1)
    for i in np.abs(Zxx):
        if max(i) > maximum:
            maximum = max(i)
    print(maximum)

    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=maximum)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.show()


# pass data as type nme.io.raw
def plot_channel_signals(data, duration, n_channels):
    # you can get the metadata included in the file and a list of all channels:
    data.plot(n_channels=n_channels, duration=duration, scalings=.0004, title='Auto-scaled Data from arrays', show=True, block=True)


def sns_diff(file):
    raw = data.get_data()
    seizure_start = seizures[file][0][0]
    seizures_end = seizures[file][0][1]
    non_seizure = raw[:, 0:seizure_start * 256] ** 2
    seizure = raw[:, seizure_start*256: seizures_end*256] ** 2
    non_seizure2 = raw[:, seizures_end*256:] ** 2
    non_seizure = np.hstack((non_seizure, non_seizure2))
    non_seizure = np.sum(non_seizure, axis=1) / (seizure_start * 256 + (len(raw[0])-seizures_end*256))
    seizure = np.sum(seizure, axis=1) / (seizures_end * 256 - seizure_start* 256)
    diff = np.abs(seizure - non_seizure)
    ch_names = data.ch_names
    print(ch_names)
    res = []
    for i in range(len(diff)):
        res.append((diff[i], ch_names[i]))

    list.sort(res)
    list.reverse(res)
    print(res)
    with open(file[:-4]+'.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in res:
            csv_writer.writerow([item[1], item[0]])

file = "/Users/reza/Downloads/Patient20/chb20_68.edf"
data = mne.io.read_raw_edf(file, preload=True)
data.pick_channels(['P4-O2', 'T8-P8-1', 'C4-P4', 'F4-C4', 'F8-T8'])
# data.pick_channels(['T7-FT9', 'T7-P7'])
raw = data.get_data()
#plot_stft(raw)
#sns_diff(file[-12:])
plot_channel_signals(data, 3000, 8)
plot_plv_vectorized()



# plot_channel_signals(data, 3500, 1)
# temp = data.crop(100, 200)

# # you can get the metadata included in the file and a list of all channels:
# info = data.info
# channels = data.ch_names
#
# data.plot(n_channels=1, duration=3500, scalings='auto', title='Auto-scaled Data from arrays', show=True, block=True)

data.pick_channels(['FZ-CZ', 'P3-O1'])
data.filter(0,2)
sig1 = data.get_data()[0]
sig2 = data.get_data()[1]
sig1 = signal.hilbert(sig1)
sig2 = signal.hilbert(sig2)
sig1 = np.angle(sig1)
sig2 = np.angle(sig2)
pld = np.unwrap(sig2) - np.unwrap(sig1)
# pld = np.exp(np.complex(0, 1) * (pld))


# List to hold x values.
x_number_values = [i / 256 for i in range(len(pld))]

# Plot the number in the list and set the line thickness.
plt.plot(x_number_values, pld, linewidth=1)

# Set the line chart title and the text font size.
plt.title("Square Numbers", fontsize=19)

# Set x axes label.
plt.xlabel("Number Value", fontsize=10)

# Set y axes label.
plt.ylabel("Square of Number", fontsize=10)

# Set the x, y axis tick marks text size.
plt.tick_params(axis='both', labelsize=9)

# Display the plot in the matplotlib's viewer.
#plt.show()