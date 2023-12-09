import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
from scipy import signal



# Definisanje parametara kanala
L = 0.5 * 1e-3  # H/km
C = 0.04 * 1e-6  # F/km
d = 1.2  # km
k = 0.18

# Frekvencijski raspon od  0 do 500 kHz
f = 500*1e3
step = 500_000
frequency_range = np.linspace(1, f, step)

# Transfer funkcija y(f)
y = 1j * 2 * np.pi * frequency_range * np.sqrt((L * C) * (1 + ((1 - 1j) * k) / (L * np.sqrt(2 * np.pi * frequency_range))))

# Frekventna karakteristika H(f)
H = np.exp(-d * y)

# Amplitudna karakteristika
amplitude = np.abs(H)

# Pretvaranje u decibele
amplitude_dB = 20 * np.log10(amplitude)

# Impulsni odziv koristeći inverznu Fourierovu transformaciju
h = np.fft.ifft(H)

# Time array za plotanje impulsnog odziva
fs = 2*f
t1 = np.linspace(0, step/fs*1e6, step)

# Amplitudna karakteristika u dB 
plt.figure(figsize=(8, 4))
plt.plot(frequency_range/1000, (amplitude_dB))
plt.title('Amplitudna karakteristika kanala')
plt.xlabel('Frekvencija (kHz)')
plt.ylabel('Amplituda (dB)')    
plt.grid(True)
plt.show()

# Impulsni odziv
plt.figure(figsize=(8, 4))
plt.plot(t1[t1 <= 50], np.real(h[t1 <= 50]))
plt.title('Impulsni odziv (0-50 μs)')
plt.xlabel('t (μs)')
plt.ylabel('h(t)')
plt.grid(True)
plt.show()


###################################################################################################

# Generisanje slučajnog signala
def generisi_signal(t, fm):
    Nsf = 0
    while Nsf < 1:
        Nsf = round(random() * 32)
    
    x = np.zeros(len(t))
    
    for _ in range(Nsf):
        znak = 1
        if random() < 0.5:
            znak = -1
        
        r_fm = round(random() * fm)
        r_theta = random() * np.pi
        r_A = random()
        
        x = x + znak * r_A * np.cos(2 * np.pi * r_fm * t + r_theta)
    
    return x


##################################################################################################

# Parametri za generisanje signala
fm = 500  # Maksimalna frekvencija u spektru signala
Bg = 400  # Širina zastitnog pojasa za frekvencijsko multipleksiranje

# Vremenski vektor
delta_t = 0.1 * 1e-6
t_max = 10 * 1e-3
num_samples = int(t_max / delta_t)
t = np.linspace(0.0000000001, t_max, num_samples)

# Frekvencijsko multipleksiranje
x_t = np.zeros_like(t)

for k in range(32):
    carrier_frequency = (fm + Bg) * k
    x_t += generisi_signal(t, fm) * np.cos(2 * np.pi * carrier_frequency * t)

# Prikaz multipleksiranog signala u vremenskom domenu
plt.figure(figsize=(10, 4))
plt.plot(t*1e3 , x_t)
plt.title('Multipleksirani Signal u Vremenskom Domenu')
plt.xlabel('t (ms)')
plt.ylabel('x (t)')
plt.ylim(-20, 20)  
plt.grid(True)
plt.show()


# Furierova transformacija x_t
x_f = np.fft.fft(x_t)

# Generisanje frekvencija
x_f_freq = np.fft.fftfreq(num_samples, delta_t)
temp = x_f_freq
x_f_freq = x_f_freq[x_f_freq >= 0]
x_f = x_f[temp >= 0]

# Amplitudni spektar x_f
plt.plot(x_f_freq / 1000, 2*np.abs(x_f)/len(x_t))
plt.xlim(0, 50)
plt.xlabel('f (kHz)')
plt.ylabel('|X(f)| dB')
plt.grid(True)
plt.show()


# Frekvencija nosioca za AM modulaciju
carrier_frequency_am = 200e3  

# AM modulacija
y_t = (1 + x_t) * np.cos(2 * np.pi * carrier_frequency_am * t)

# Prikaz signala u vremenskom domenu
plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, y_t)
plt.title('AM Modulisani Signal u Vremenskom Domenu')
plt.xlabel('t (ms)')
plt.ylabel('y(t)')
plt.ylim(-20, 20)
plt.grid(True)
plt.show()


# Furierova transformacija za y_t
y_f = np.fft.fft(y_t)

# Generisanje frekvencija za y_t
y_f_freq = np.fft.fftfreq(num_samples, delta_t)

# Amplitudni spektar za y_f
plt.plot(y_f_freq / 1000, 2*np.abs(y_f)/len(y_t))
plt.xlim(0, 500)  
plt.xlabel('f (kHz)')
plt.ylabel('|Y(f)| dB')
plt.ylim(0,2)
plt.grid(True)
plt.title('Amplitudni Spektar AM Moduliranog Signala')
plt.show()

# Konvolucija za uticaj kanala na signal
z_t = np.convolve(np.real(h), y_t)
z_t = z_t[:len(y_t)]

# Furierova transformacija za konvoluirani signal
z_f = np.fft.fft(z_t)

# Generisanje frekvencija za z_t
z_f_freq = np.fft.fftfreq(len(z_t), delta_t)

# Amplitudni spektar konvoluiranog signala
plt.plot(z_f_freq / 1000, 2 * np.abs(z_f) / len(z_t))
plt.xlim(0, 500) 
plt.xlabel('f (kHz)')
plt.ylabel('|Z(f)| dB')
plt.ylim(0, 2)
plt.grid(True)
plt.title('Amplitudni Spektar Konvoluiranog Signala')
plt.show()

# Prikaz signala u vremenskom domenu nakon konvolucije
plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, np.real(z_t))
plt.title('Vremenski Odziv Nakon Konvolucije')
plt.xlabel('t (ms)')
plt.ylabel('z(t)')
plt.ylim(-20, 20)
plt.grid(True)
plt.show()


# Frekvencija nosioca za demodulaciju
carrier_frequency_demod = 200e3

# Demodulacija
w_t = z_t * np.cos(2 * np.pi * carrier_frequency_demod * t)

# Prikaz demoduliranog signala
plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, w_t)
plt.title('Demodulirani Signal')
plt.xlabel('t (ms)')
plt.ylabel('w(t)')
plt.grid(True)
plt.show()

# Furierova transformacija demoduliranog signala
w_f = np.fft.fft(w_t)

# Generisanje frekvencija za w_t
w_f_freq = np.fft.fftfreq(len(w_t), delta_t)

# Amplitudni spektar demoduliranog signala
plt.plot(w_f_freq / 1000, 2 * np.abs(w_f) / len(w_t))
plt.xlim(0, 500)  
plt.xlabel('f (kHz)')
plt.ylabel('|W(f)| dB')
plt.grid(True)
plt.title('Amplitudni Spektar Demoduliranog Signala')
plt.show()


# demultipleksiranje
demultiplexed_signals = []

for k in range(32):
    carrier_frequency_demux = (fm + Bg) * k
    demux_signal = w_t * np.sin(2 * np.pi * carrier_frequency_demux * t)
    demultiplexed_signals.append(demux_signal)






# Koeficijenti za niskopropusni filter
numerator_coeffs_lp = [0.0952, 0.4760, 0.9520, 0.4760, 0.0952]
denominator_coeffs_lp = [1.0000, -4.9898, 9.9594, -9.9392, 4.9595, -0.9899]

# Funkcija za zamjenu NaN vrijednosti sa 0 u signalu
def zamijeni_nan_sa_nulom(signal_to_process):
    signal_to_process[np.isnan(signal_to_process)] = 0
    return signal_to_process

# Funkcija za primjenu filtfilt na svaki element u listi
def primijeni_filter(signal_to_filter):
    filtrirani_signal = signal.filtfilt(numerator_coeffs_lp, denominator_coeffs_lp, signal_to_filter)
    return filtrirani_signal





# Zamjena NaN vrijednosti sa 0 u svakom elementu demultiplexed_signals
demultiplexed_signals_bez_nan = [zamijeni_nan_sa_nulom(signal_element) for signal_element in demultiplexed_signals]

# Primjena filtra na svaki element u demultiplexed_signals_bez_nan
filtrirani_signali = [primijeni_filter(signal_element) for signal_element in demultiplexed_signals_bez_nan]
# Filtriranje ne uspjeva iz nekog razloga!!!!


# Prikaz originalnog i filtriranog signala za jedan od elemenata
plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, demultiplexed_signals[1], label='Originalni Signal')
plt.plot(t * 1e3, filtrirani_signali[1], label='Filtrirani Signal')
plt.title('Originalni i Filtrirani Signal')
plt.xlabel('t (ms)')
plt.ylabel('Amplituda Signala')
plt.legend()
plt.grid(True)
plt.show()
