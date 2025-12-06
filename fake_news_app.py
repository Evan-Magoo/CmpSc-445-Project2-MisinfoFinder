import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import Image, ImageTk
from newspaper import Article
from wordcloud import WordCloud
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

window = tk.Tk()
window.title('Abington Evacuation Tool')
window.geometry('900x700')
window.configure(bg='#001E44')
logo = tk.PhotoImage(file="favicon.png")
window.iconphoto(False, logo)

def predict_news(link):
    global output_box
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

def fake_news_screen():
    global output_box
    clear_screen()

    title_label = tk.Label(
        window,
        text="Fake News Detection",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    link = tk.StringVar()

    controls = tk.Frame(window, bg="#001E44")
    controls.pack(pady=10)

    entry_label = tk.Label(
        controls, 
        text='Article Link:', 
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

    screen_selection = tk.Frame(window, bg="#001E44")
    screen_selection.pack(pady=10, side=tk.BOTTOM)

    fake_news_button = tk.Button(
        screen_selection,
        text="Fake News Detection",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_screen
    )

    fake_classifciation_button = tk.Button(
        screen_selection,
        text="Fake News Classification",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_classification_screen
    )

    article_word_cloud_button = tk.Button(
        screen_selection,
        text="Article Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_word_cloud_screen
    )

    article_summarizer_button = tk.Button(
        screen_selection,
        text="Article Summarizer",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_summarizer_screen
    )

    fake_news_button.grid(row=0, column=0, padx=10)
    fake_classifciation_button.grid(row=0, column=1, padx=10)
    article_word_cloud_button.grid(row=0, column=2, padx=10)
    article_summarizer_button.grid(row=0, column=3, padx=10)

def fake_news_classification_screen():
    clear_screen()
    title_label = tk.Label(
        window,
        text="Fake News Classification",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    screen_selection = tk.Frame(window, bg="#001E44")
    screen_selection.pack(pady=10, side=tk.BOTTOM)

    fake_news_button = tk.Button(
        screen_selection,
        text="Fake News Detection",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_screen
    )

    fake_classifciation_button = tk.Button(
        screen_selection,
        text="Fake News Classification",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_classification_screen
    )

    article_word_cloud_button = tk.Button(
        screen_selection,
        text="Article Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_word_cloud_screen
    )

    article_summarizer_button = tk.Button(  
        screen_selection,
        text="Article Summarizer",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_summarizer_screen
    )

    fake_news_button.grid(row=0, column=0, padx=10)
    fake_classifciation_button.grid(row=0, column=1, padx=10)
    article_word_cloud_button.grid(row=0, column=2, padx=10)
    article_summarizer_button.grid(row=0, column=3, padx=10)

def article_word_cloud_screen():
    clear_screen()
    title_label = tk.Label(
        window,
        text="Article Word Cloud",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    link = tk.StringVar()

    controls = tk.Frame(window, bg="#001E44")
    controls.pack(pady=10)

    entry_label = tk.Label(
        controls, 
        text='Article Link:', 
        font=('Arial', 12, 'bold'),
        fg='white',
        background='#001E44'
    )

    entry = tk.Entry(
        controls,
        textvariable=link,
        width = 50,
        font=("Arial", 11)
    )

    entry_label.grid(row=0, column=0, padx=5)
    entry.grid(row=0, column=1, padx=5)

    def generate_word_cloud():
        try:
            article = Article(link.get())
            article.download()
            article.parse()
            text = article.text
            wc = WordCloud(width=827, height=450, background_color='white').generate(text)
            img = wc.to_image()
            img_tk = ImageTk.PhotoImage(img)

            if hasattr(article_word_cloud_screen, 'img_label') and article_word_cloud_screen.img_label.winfo_exists():
                article_word_cloud_screen.img_label.configure(image=img_tk)
                article_word_cloud_screen.img_label.image = img_tk
            else:
                article_word_cloud_screen.img_label = tk.Label(window, image=img_tk)
                article_word_cloud_screen.img_label.image = img_tk
                article_word_cloud_screen.img_label.pack(pady=20)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    generate_button = tk.Button(
        controls,  
        text="Generate Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=generate_word_cloud
    )
    generate_button.grid(row=0, column=2, padx=5)

    screen_selection = tk.Frame(window, bg="#001E44")
    screen_selection.pack(pady=10, side=tk.BOTTOM)

    fake_news_button = tk.Button(
        screen_selection,
        text="Fake News Detection",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_screen
    )

    fake_classifciation_button = tk.Button(
        screen_selection,
        text="Fake News Classification",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_classification_screen
    )

    article_word_cloud_button = tk.Button(
        screen_selection,
        text="Article Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_word_cloud_screen
    )

    article_summarizer_button = tk.Button(
        screen_selection,
        text="Article Summarizer",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_summarizer_screen
    )

    fake_news_button.grid(row=0, column=0, padx=10)
    fake_classifciation_button.grid(row=0, column=1, padx=10)
    article_word_cloud_button.grid(row=0, column=2, padx=10)
    article_summarizer_button.grid(row=0, column=3, padx=10)

def article_summarizer_screen():
    clear_screen()
    title_label = tk.Label(
        window,
        text="Article Summarizer",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    screen_selection = tk.Frame(window, bg="#001E44")
    screen_selection.pack(pady=10, side=tk.BOTTOM)

    fake_news_button = tk.Button(
        screen_selection,
        text="Fake News Detection",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_screen
    )

    fake_classifciation_button = tk.Button(
        screen_selection,
        text="Fake News Classification",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_classification_screen
    )

    article_word_cloud_button = tk.Button(
        screen_selection,
        text="Article Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_word_cloud_screen
    )

    article_summarizer_button = tk.Button(
        screen_selection,
        text="Article Summarizer",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_summarizer_screen
    )

    fake_news_button.grid(row=0, column=0, padx=10)
    fake_classifciation_button.grid(row=0, column=1, padx=10)
    article_word_cloud_button.grid(row=0, column=2, padx=10)
    article_summarizer_button.grid(row=0, column=3, padx=10)


if __name__ == "__main__":
    clear_screen()
    fake_news_screen()
    window.mainloop()