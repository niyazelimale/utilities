import customtkinter as ctk
from tkinter import filedialog, messagebox
import ollama
from docx import Document
from xhtml2pdf import pisa
import html

# Set appearance and theme globally
ctk.set_appearance_mode("dark")  # options: "dark", "light", or "system"
ctk.set_default_color_theme("dark-blue")  # options: "blue", "dark-blue", "green"

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GenAI Chat (Ollama Llama 3.1 8B)")
        self.geometry("720x600")
        self.messages = []
        self.last_ai_response = ""

        self.create_widgets()

    def create_widgets(self):
        # Chat display (read-only)
        self.chat_display = ctk.CTkTextbox(self, width=680, height=460, wrap="word")
        self.chat_display.configure(state="disabled")
        self.chat_display.pack(padx=20, pady=(20, 10))

        # Input frame for prompt entry and buttons
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Input entry with placeholder text
        self.entry = ctk.CTkEntry(input_frame, placeholder_text="Enter your prompt here...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=10)
        self.entry.bind("<Return>", lambda event: self.send_message())

        # Send button
        send_btn = ctk.CTkButton(input_frame, text="Send", width=80, command=self.send_message)
        send_btn.pack(side="left", padx=(0, 5), pady=10)

        # Save response button
        save_btn = ctk.CTkButton(input_frame, text="Save Response", width=120, command=self.save_response)
        save_btn.pack(side="left", pady=10)

    def send_message(self):
        user_msg = self.entry.get().strip()
        if not user_msg:
            return
        self.entry.delete(0, ctk.END)

        # Show user message
        self.append_chat(f"You: {user_msg}\n")
        self.messages.append({"role": "user", "content": user_msg})

        try:
            # Chat with Ollama Llama 3.1 8B model
            response = ollama.chat(model="llama3.1:8b", messages=self.messages)
            ai_msg = response["message"]["content"]
            self.messages.append({"role": "assistant", "content": ai_msg})

            self.append_chat(f"AI: {ai_msg}\n\n")
            self.last_ai_response = ai_msg

        except Exception as e:
            messagebox.showerror("AI Error", f"Failed to get response from AI:\n{e}")

    def append_chat(self, text):
        self.chat_display.configure(state="normal")
        self.chat_display.insert(ctk.END, text)
        self.chat_display.see(ctk.END)
        self.chat_display.configure(state="disabled")

    def save_response(self):
        if not self.last_ai_response:
            messagebox.showerror("No Response", "No AI response to save yet.")
            return

        save_file = filedialog.asksaveasfile(
            mode='w',
            defaultextension='.docx',
            filetypes=[
                ("Word Document", "*.docx"),
                ("PDF File", "*.pdf"),
                ("HTML File", "*.html"),
            ]
        )
        if not save_file:
            return

        filename = save_file.name
        save_file.close()

        try:
            if filename.endswith('.docx'):
                doc = Document()
                doc.add_paragraph(self.last_ai_response)
                doc.save(filename)

            elif filename.endswith('.html'):
                html_content = f"<html><body><pre>{html.escape(self.last_ai_response)}</pre></body></html>"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

            elif filename.endswith('.pdf'):
                html_content = f"<html><body><pre>{html.escape(self.last_ai_response)}</pre></body></html>"
                with open(filename, 'wb') as pdf_file:
                    pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
                    if pisa_status.err:
                        raise Exception("PDF generation failed")

            else:
                messagebox.showerror("Unsupported Format", "Please save as DOCX, PDF, or HTML.")
                return

            messagebox.showinfo("Saved", f"Response saved successfully to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file:\n{e}")

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
