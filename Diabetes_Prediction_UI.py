import tkinter as tk
import Diabetes_module
window=tk.Tk()

window.title("Diabetes Prediction")
window.geometry("800x700")

#--display function to button

def display():
	#.get() gets the input value from the user and return a string type
	name = E_name.get()
	age = (int)(E_age.get())
	bp = (float)(E_bp.get())
	bmi = (float)(E_bmi.get())
	predictor =Diabetes_module.predict(age,bmi,bp)
	msg = "Hello, "+str(name)+"\nyour details are:\n"+"age :"+str(age)+"\nBlood Pressure: "+str(bp)+"\nBMI : "+str(bmi)+"\nPredicted value : "+str(predictor)
	if(predictor==1):
		msg=msg+"\n\nwell ,nothing to worry about ,though you have type-2 Diabetes, please consult your physician for your diet and enjoy happy living"
	else:
		msg=msg+"\n\n you don't have type-2 Diabetes"
	info_display = tk.Text(master = window,height = 10,width = 50,font=("Gadugi",13,"bold"))
	info_display.grid(column=0,row=10)

	info_display.insert(tk.END,msg)


#-----Labels------
#--title---
myTitle = tk.Label(text="Diabetes Prediction",font=("Algerian",15,"bold"))

myTitle.grid(column=0,row=0)

L_name = tk.Label(text="Name",font=("Sitka Subheading",12))
L_name.grid(column=0,row=2)

E_name = tk.Entry()
E_name.grid(column=1,row=2)

L_age=tk.Label(text="Age",font=("Sitka Subheading",12))
L_age.grid(column=0,row=3)

E_age = tk.Entry()
E_age.grid(column=1,row=3)

L_bp = tk.Label(text="Blood Pressure",font=("Sitka Subheading",12))
L_bp.grid(column=0,row=4)

E_bp = tk.Entry()
E_bp.grid(column=1,row=4)

L_bmi = tk.Label(text="BMI",font=("Sitka Subheading",12))
L_bmi.grid(column=0,row=6)

E_bmi = tk.Entry()
E_bmi.grid(column=1,row=6)

#--Button--
#--command is to give an action to button
B_predict = tk.Button(text="PREDICT",font=("Sitka Subheading",12),command=display)
B_predict.grid(column=1,row=8)
#

window.mainloop()