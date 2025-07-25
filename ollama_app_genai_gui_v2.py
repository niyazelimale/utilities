import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import ollama
from docx import Document
from xhtml2pdf import pisa

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GenAI Chat (Ollama Llama 3.1 8B)")
        self.messages = []
        self.last_ai_response = ""
        self.create_widgets()

    def create_widgets(self):
        # Chat display area (read-only)
        self.chat_display = scrolledtext.ScrolledText(self.root, width=70, height=25, state='disabled')
        self.chat_display.pack(pady=10, padx=10)

        # Input frame for prompt entry and buttons
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill='x', padx=10, pady=(0,10))

        # Text entry for user prompt
        self.entry = tk.Entry(input_frame, width=60)
        self.entry.pack(side=tk.LEFT, fill='x', expand=True)
        self.entry.bind('<Return>', lambda event: self.send_message())

        # Send button
        send_btn = tk.Button(input_frame, text="Send", command=self.send_message)
        send_btn.pack(side=tk.LEFT, padx=5)

        # Save response button
        save_btn = tk.Button(input_frame, text="Save Response", command=self.save_response)
        save_btn.pack(side=tk.LEFT, padx=5)

    def send_message(self):
        user_msg = self.entry.get().strip()
        if not user_msg:
            return
        self.entry.delete(0, tk.END)

        # Show user message in chat display
        self.update_chat(f"You: {user_msg}\n", "user")
        self.messages.append({"role": "user", "content": user_msg})

        try:
            # Send conversation history to Ollama model and get response
            response = ollama.chat(model="llama3.1:8b", messages=self.messages)
            ai_msg = response["message"]["content"]
            self.messages.append({"role": "assistant", "content": ai_msg})

            # Show AI message
            self.update_chat(f"AI: {ai_msg}\n\n", "assistant")

            # Save latest AI message for saving to file
            self.last_ai_response = ai_msg
        except Exception as e:
            messagebox.showerror("Error", f"AI Error: {e}")

    def update_chat(self, text, sender):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

    def save_response(self):
        if not self.last_ai_response:
            messagebox.showerror("Error", "No AI response to save yet.")
            return

        file = filedialog.asksaveasfile(
            mode='w',
            defaultextension='.docx',
            filetypes=[("Word Document", "*.docx"),
                       ("PDF File", "*.pdf"),
                       ("HTML File", "*.html")]
        )
        if not file:
            return

        filename = file.name
        file.close()  # Close immediately to use different writing methods below

        try:
            if filename.endswith('.docx'):
                doc = Document()
                doc.add_paragraph(self.last_ai_response)
                doc.save(filename)

            elif filename.endswith('.html'):
                html_content = f"<html><body><pre>{self.escape_html(self.last_ai_response)}</pre></body></html>"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

            elif filename.endswith('.pdf'):
                html_content = f"<html><body><pre>{self.escape_html(self.last_ai_response)}</pre></body></html>"
                with open(filename, 'wb') as pdf_file:
                    pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
                    if pisa_status.err:
                        raise Exception("PDF creation failed")

            else:
                messagebox.showerror("Error", "Unsupported file format.")
                return

            messagebox.showinfo("Success", f"Response saved successfully as:\n{filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save the file: {e}")

    @staticmethod
    def escape_html(text):
        # Simple HTML escaping
        import html
        return html.escape(text)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
