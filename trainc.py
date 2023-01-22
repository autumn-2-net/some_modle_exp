from diffwave.__main__ import mymain

class t_obj:
    model_dir='./md/'
    data_dirs=['./testwav/']
    max_steps=10000
    fp16=False

ttt=t_obj()



if __name__ == "__main__":
    mymain(ttt)