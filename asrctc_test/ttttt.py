import soundfile
import sox
import os
sr = 16000

tfm = sox.Transformer()
tfm.speed(2)





array_out = tfm.build_array(input_filepath="2099003695.wav")
soundfile.write("20990036951.wav",data=array_out,samplerate=sr)

NO_SOX = False
stream_handler = os.popen('sox -h')
aaa=stream_handler.readlines()
if not len(stream_handler.readlines()):

    NO_SOX = True
stream_handler.close()
