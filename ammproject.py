import glob 
import cv2
from gtts import gTTS
import IPython.display as ipd

sift = cv2.SIFT_create()

index_params = dict(algorithm=0, trees=5)
search_params = dict()
bf = cv2.FlannBasedMatcher(index_params, search_params)

dic={1:0,5:0,10:0,20:0,50:0,100:0,200:0}

def sizeing(image,x,y):
    return  cv2.resize(image,(x, y))

def image_gray(image):
    Imagecolor=cv2.imread(image)#read image path ,flag=0:grayscale 1=color -1=alpha 
    Imagegray = cv2.cvtColor(Imagecolor, cv2.COLOR_BGR2GRAY)
    return Imagegray

def desc(des_of_images):
    descriptor={}
    for key, value in des_of_images.items():
        kp,des=sift.detectAndCompute(value,None)
        descriptor[key]=des
    return descriptor

def knnMatch(des1,descriptor):
    one_verse_all={}
    for key,value in descriptor.items():
        one_verse_all[key]=bf.knnMatch(des1,descriptor[key],k=2)
    return one_verse_all

def valuess (one_verse_all):
    values={}
    for key, value in one_verse_all.items():
        good = []
        for m,n in value:
             if m.distance < .9*n.distance:
                good.append([m])
        values[key]=len(good)
    return values

def value(values):
    Keymax= max(values, key=values.get)
    return ((int)(Keymax.split(' ')[0]))

def total(dic):
    tot=0
    for i,j in dic.items():
        if (i*j!=0):
            tot+=i*j
    return tot

def cout(dic):
    string='In pocket,'
    for i,j in dic.items():
        string+=' '+str(j)+' of '+str(i)+','
    string+=" and the total is "+str(total(dic))
    return string

def texttoaudio(dic):
    language = 'en'
    myobj = gTTS(text=dic, lang=language, slow=False)   
    myobj.save("message.mp3")
    return "message.mp3"

def playAudio(txt):
    return ipd.Audio(txt, autoplay=True)

def func(image,des_of_images):
    Imagecolor=sizeing(image_gray(image),1200,500)
    kp1, des1 = sift.detectAndCompute(Imagecolor,None)
    descriptor=desc(des_of_images)
    one_verse_all=knnMatch(des1,descriptor)
    values=valuess(one_verse_all)
    dic[value(values)]=dic[value(values)]+1


des_of_images={}
src={}
i=0
for img in glob.glob("new_dataset/Training/*.jpg"):
    Imagegray =image_gray(img)
    Imagegray = sizeing(Imagegray,1200,500)
    img=(img.split("\\")[1]).split(".")[0]
    des_of_images[img]=Imagegray
    src[i]=img
    i=i+1
print(src)

"""
des_of_images={}
src={}
i=0
for img in glob.glob("images_training/*.jpg"):
    Imagegray =image_gray(img)
    Imagegray = sizeing(Imagegray,1200,500)
    img=(img.split("\\")[1]).split(".")[0]
    des_of_images[img]=Imagegray
    src[i]=img
    i=i+1
print(src)
"""

func("new_dataset/Testing/1 back.jpg",des_of_images)
func("new_dataset/Testing/1 front.jpg",des_of_images)
func("new_dataset/Testing/5 back.jpg",des_of_images)
func("new_dataset/Testing/5 front.jpg",des_of_images)
func("new_dataset/Testing/10 back.jpg",des_of_images)
func("new_dataset/Testing/10 front.jpg",des_of_images)
func("new_dataset/Testing/10 backn.jpg",des_of_images)
func("new_dataset/Testing/10 frontn.jpg",des_of_images)
func("new_dataset/Testing/20 back.jpg",des_of_images)
func("new_dataset/Testing/20 front.jpg",des_of_images)
func("new_dataset/Testing/50 back.jpg",des_of_images)
func("new_dataset/Testing/50 front.jpg",des_of_images)
func("new_dataset/Testing/100 back.jpg",des_of_images)
func("new_dataset/Testing/100 front.jpg",des_of_images)
func("new_dataset/Testing/200 back.jpg",des_of_images)
func("new_dataset/Testing/200 front.jpg",des_of_images)

"""
func("images_testing/10 back.jpg",des_of_images)
func("images_testing/10 front.jpg",des_of_images)
func("images_testing/20 back.jpg",des_of_images)
func("images_testing/20 front.jpg",des_of_images)
func("images_testing/50 back.jpg",des_of_images)
func("images_testing/50 front.jpg",des_of_images)
func("images_testing/100 back.jpg",des_of_images)
func("images_testing/100 front.jpg",des_of_images)
func("images_testing/200 back.jpg",des_of_images)
func("images_testing/200 front.jpg",des_of_images)
"""
"""
func("20 new.jpg",des_of_images)
func("20 newf.jpg",des_of_images)
"""

print(dic)
print(cout(dic))

string=cout(dic)
string2=texttoaudio(string)
playAudio(string2)
