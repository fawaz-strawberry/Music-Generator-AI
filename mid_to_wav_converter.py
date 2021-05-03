'''
So what needs to be done

I need to generate an output wav for each midi file. 
And for each output wav I need to break it up into 10 seconds files.
Then convert each of those 10 second files into a spectogram with the
label of X_Y.png where X = the song number and Y = the segment order.

'''

import numpy as np
import math
from PIL import Image
import time

import scipy.io.wavfile
rate,audData=scipy.io.wavfile.read("never-gonna-give-you-up.wav")
audData.shape[0] / rate / 60

channel1=audData[:,0] #left
channel2=audData[:,1] #right
signal_fragment = channel1
FFT_LENGTH = 1024
WINDOW_LENGTH = 1024
WINDOW_STEP = int(WINDOW_LENGTH / 2)
magnitudeMin = float("inf")
magnitudeMax = float("-inf")
phaseMin = float("inf")
phaseMax = float("-inf")


def amplifyMagnitudeByLog(d):
    return 188.301 * math.log10(d + 1)
    #return 200 * math.log10(d+1)

def weakenAmplifiedMagnitude(d):
    return math.pow(10, d/188.301)-1
    #return math.pow(10, d/300)-1

def generateLinearScale(magnitudePixels, phasePixels, 
                        magnitudeMin, magnitudeMax, phaseMin, phaseMax):
    height = magnitudePixels.shape[0]
    width = magnitudePixels.shape[1]
    magnitudeRange = magnitudeMax - magnitudeMin
    phaseRange = phaseMax - phaseMin
    rgbArray = np.zeros((height, width, 3), 'uint8')
    
    for w in range(width):
        for h in range(height):
            magnitudePixels[h,w] = (magnitudePixels[h,w] - magnitudeMin) / (magnitudeRange) * 255 * 2
            magnitudePixels[h,w] = amplifyMagnitudeByLog(magnitudePixels[h,w])
            phasePixels[h,w] = (phasePixels[h,w] - phaseMin) / (phaseRange) * 255
            red = 255 if magnitudePixels[h,w] > 255 else magnitudePixels[h,w]
            green = (magnitudePixels[h,w] - 255) if magnitudePixels[h,w] > 255 else 0
            blue = phasePixels[h,w]
            rgbArray[h,w,0] = int(red)
            rgbArray[h,w,1] = int(green)
            rgbArray[h,w,2] = int(blue)
    return rgbArray

def recoverLinearScale(rgbArray, magnitudeMin, magnitudeMax, 
                       phaseMin, phaseMax):
    width = rgbArray.shape[1]
    height = rgbArray.shape[0]

    magnitudeVals = rgbArray[:,:,0].astype(float) + rgbArray[:,:,1].astype(float)
    
    phaseVals = rgbArray[:,:,2].astype(float)
    
    phaseRange = phaseMax - phaseMin
    magnitudeRange = magnitudeMax - magnitudeMin



    for w in range(width):
        for h in range(height):
            phaseVals[h,w] = (phaseVals[h,w] / 255 * phaseRange) + phaseMin
            magnitudeVals[h,w] = weakenAmplifiedMagnitude(magnitudeVals[h,w])
            magnitudeVals[h,w] = (magnitudeVals[h,w] / (255*2) * magnitudeRange) + magnitudeMin

    return magnitudeVals, phaseVals

def generateSpectrogramForWave(signal):
    start_time = time.time()
    global magnitudeMin
    global magnitudeMax
    global phaseMin
    global phaseMax
    buffer = np.zeros(int(signal.size + WINDOW_STEP - (signal.size % WINDOW_STEP)))
    buffer[0:len(signal)] = signal
    height = int(FFT_LENGTH / 2 + 1)
    width = int(len(buffer) / (WINDOW_STEP) - 1)
    magnitudePixels = np.zeros((height, width))
    phasePixels = np.zeros((height, width))

    for w in range(width):
        buff = np.zeros(FFT_LENGTH)
        stepBuff = buffer[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH]
        # apply hanning window
        stepBuff = stepBuff * np.hanning(WINDOW_LENGTH)
        buff[0:len(stepBuff)] = stepBuff
        #buff now contains windowed signal with step length and padded with zeroes to the end
        fft = np.fft.rfft(buff)
        for h in range(len(fft)):
            #print(buff.shape)
            #return
            magnitude = math.sqrt(fft[h].real**2 + fft[h].imag**2)
            if magnitude > magnitudeMax:
                magnitudeMax = magnitude 
            if magnitude < magnitudeMin:
                magnitudeMin = magnitude 

            phase = math.atan2(fft[h].imag, fft[h].real)
            if phase > phaseMax:
                phaseMax = phase
            if phase < phaseMin:
                phaseMin = phase
            magnitudePixels[height-h-1,w] = magnitude
            phasePixels[height-h-1,w] = phase
    rgbArray = generateLinearScale(magnitudePixels, phasePixels,
                                  magnitudeMin, magnitudeMax, phaseMin, phaseMax)
    elapsed_time = time.time() - start_time
    print('%.2f' % elapsed_time, 's', sep='')
    img = Image.fromarray(rgbArray, 'RGB')
    return img



def recoverSignalFromSpectrogram(filePath, track_id):
    img = Image.open(filePath)
    data = np.array( img, dtype='uint8' )
    #print(data)
    width = data.shape[1]
    height = data.shape[0]

    magnitudeVals, phaseVals \
    = recoverLinearScale(data, magnitudeMin, magnitudeMax, phaseMin, phaseMax)
    recovered = np.zeros(WINDOW_LENGTH * width // 2 + WINDOW_STEP, dtype=np.int16)
    for w in range(width):
        toInverse = np.zeros(height, dtype=np.complex_)
        for h in range(height):
            magnitude = magnitudeVals[height-h-1,w]
            phase = phaseVals[height-h-1,w]
            toInverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
        signal = np.fft.irfft(toInverse)
        recovered[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH] += signal[:WINDOW_LENGTH].astype(np.int16)
    scipy.io.wavfile.write("./recovered" + str(track_id) + ".wav", rate, recovered)



from midi2audio import FluidSynth
from pydub import AudioSegment
import mutagen
from mutagen.wave import WAVE
import os



def single_run():

    audio = WAVE("never-gonna-give-you-up.wav")
    audio_info = audio.info
    song_duration = int(audio_info.length)

    for i in range(int(song_duration/5) + 1):
        t1 = i * 5000
        t2 = (i+1) * 5000
        #new_audio = AudioSegment.from_wav("output.wav")
        new_audio = AudioSegment.from_wav("never-gonna-give-you-up.wav")
        new_audio = new_audio[t1:t2]
        new_audio.export(("audio_files/" + str(i) + ".wav"), format="wav")
    
        rate,audData=scipy.io.wavfile.read('audio_files/' + str(i) + '.wav')
        audData.shape[0] / rate / 60

        channel1=audData[:,0] #left
        channel2=audData[:,1] #right

        channel_final = np.mean( np.array([ channel1, channel2 ]), axis=0 )
        signal_fragment = np.asarray(channel_final)
        
        for j in range (len(signal_fragment)):
            if(signal_fragment[j] == None):
                print("Error :/")
        
        try:
            img = generateSpectrogramForWave(signal_fragment)
            #scipy.io.wavfile.write("/input/before.wav", rate, signal_fragment)
            img.save("audio_image_test/" + str(999) + "_" + str(i) + ".png","PNG")
            # if(k % 8 == 0):
            #recoverSignalFromSpectrogram("audio_image_test/" + str(999) + "_" + str(i) + ".png", i)
        except:
            print("Error Generating Last Image, probably an even number")
        break


def recover_audio(file_path, num):
    recoverSignalFromSpectrogram(file_path, num)

def mega_run():
    #Convert midi file to .wav file
    fs = FluidSynth("Piano.SF2")

    abs_path = "/mnt/c/Users/fawaz/Music/lmd_full/lmd_full/0/"
    zero_songs = os.listdir("/mnt/c/Users/fawaz/Music/lmd_full/lmd_full/0")
    #for k in range(1):
    for k in range(len(zero_songs)):
        fs.midi_to_audio((abs_path + zero_songs[k]), 'output.wav')

        try:
            audio = WAVE("output.wav")
        except:
            print("No Output file found")
            continue

        #audio = WAVE("never-gonna-give-you-up.wav")
        audio_info = audio.info
        song_duration = int(audio_info.length)


        for i in range(int(song_duration/5) + 1):
            t1 = i * 5000
            t2 = (i+1) * 5000
            new_audio = AudioSegment.from_wav("output.wav")
            #new_audio = AudioSegment.from_wav("never-gonna-give-you-up.wav")
            new_audio = new_audio[t1:t2]
            new_audio.export(("audio_files/" + str(i) + ".wav"), format="wav")
        
            rate,audData=scipy.io.wavfile.read('audio_files/' + str(i) + '.wav')
            audData.shape[0] / rate / 60

            channel1=audData[:,0] #left
            channel2=audData[:,1] #right

            channel_final = np.mean( np.array([ channel1, channel2 ]), axis=0 )
            signal_fragment = np.asarray(channel_final)



            try:
                img = generateSpectrogramForWave(signal_fragment)
                #scipy.io.wavfile.write("/input/before.wav", rate, signal_fragment)
                img.save("audio_images/" + str(k) + "_" + str(i) + ".png","PNG")
                if(k % 8 == 0):
                    recoverSignalFromSpectrogram("audio_images/" + str(k) + "_" + str(i) + ".png", i)
            except:
                print("Error Generating Last Image, probably an even number")




# single_run()
# recover_audio("audio_image_test/" + str(999) + "_" + str(1) + ".png", 999)

single_run()

for i in range(100):
    recover_audio(("FinalGeneratedAudio/" + "THE_REAL_FINAL_OUTPUT_WAVS/output_" + str(i) + ".png"), i)

#single_run()
#recover_audio("FinalGeneratedAudio/0_0.png", 1001)