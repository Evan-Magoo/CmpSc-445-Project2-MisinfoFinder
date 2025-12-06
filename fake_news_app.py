import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from newspaper import Article
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

window = tk.Tk()
window.title('Abington Evacuation Tool')
window.geometry('900x500')
window.configure(bg='#001E44')
logo = tk.PhotoImage(file="favicon.png")
window.iconphoto(False, logo)

def predict_news(link):
    output_box.delete(1.0, tk.END)

    try:
        article = Article(link)
        article.download()
        article.parse()
        text = article.text

        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        result = "REAL" if prediction == 1 else "FAKE"

        output_box.insert(tk.END, f"Prediction: {result}\n\n")
        output_box.insert(tk.END, "---- Article Text ----\n\n")
        output_box.insert(tk.END, text)

    except Exception as e:
        output_box.insert(tk.END, f"Error fetching article:\n{e}")

def clear_screen():
    for widget in window.winfo_children():
        widget.destroy()

if __name__ == "__main__":
    clear_screen()

    link = tk.StringVar()

    controls = tk.Frame(window, bg="#001E44")
    controls.pack(pady=10)

    entry_label = tk.Label(
        controls, 
        text='Articel Link:', 
        font=('Arial', 12, 'bold'),
        fg='white',
        background='#001E44'
    )

    entry = tk.Entry(
        controls,
        textvariable=link,
        width = 50,
        font=("Arial", 12)
    )

    entry_button = tk.Button(
        controls,
        text="Enter",
        width=12,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=lambda: predict_news(entry.get())
    )

    entry_label.grid(row=0, column=0, padx=5)
    entry.grid(row=0, column=1, padx=5)
    entry_button.grid(row=0, column=2, padx=5)

    output_box = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        width=90,
        height=25,
        font=("Arial", 12)
    )
    output_box.pack(pady=20)

    window.mainloop()