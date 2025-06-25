import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import sys
import io
import os
from datetime import datetime
from typing import Dict, Any
import queue
from Code.logic.controller.RequirementsController import RequirementsController


class ConsoleRedirect:
    """Class to redirect console exit to the interface"""

    def __init__(self, text_widget, queue_obj):
        self.text_widget = text_widget
        self.queue = queue_obj

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass


class RequirementsGeneratorGUI:
    def __init__(self):
        # Configuración inicial de CustomTkinter
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # Ventana principal
        self.root = ctk.CTk()
        self.root.title("Requirements Generator - Control Panel")
        self.root.geometry("1600x900")
        self.root.resizable(True, True)

        # Variables de control
        self.controller = RequirementsController.get_instance()
        self.is_running = False

        # Queue para la consola
        self.console_queue = queue.Queue()

        self._setup_variables()
        self._create_widgets()
        self._setup_console_redirect()
        self._start_console_update()

    def _setup_variables(self):
        """Config controls variables for parameters"""

        # Rutas de archivos
        self.csv_path = ctk.StringVar(value="")
        self.output_directory = ctk.StringVar(value="")
        self.app_description = ctk.StringVar(value="An app named SwiftKey that offers an alternative keyboard for your smartphone")

        # Parámetros de filtrado
        self.classification_model = ctk.StringVar(value="BERTweet - base")
        self.vector_type = ctk.StringVar(value="RC")

        # Parámetros de clustering
        self.clustering_algorithm = ctk.StringVar(value="fuzzy c-means")
        self.dim_reduction = ctk.StringVar(value="UMAP")
        self.k_min = ctk.IntVar(value=3)
        self.k_max = ctk.IntVar(value=10)

        # Parámetros de generación
        self.llm_model = ctk.StringVar(value="deepseek-ai/DeepSeek-V3-0324")
        self.llm_provider = ctk.StringVar(value="fireworks-ai")

        # Opciones predefinidas (fácil de expandir)
        self.classification_models = [
            "BERTweet - base",
            "RoBERTa - base",
            "XLNet - base",
            "ALBERT v1 - large",
            "GPT 2"
        ]

        self.vector_types = [
            "None",
            "Relevant Count (RC)",
            "Relevant Position (RP)"
        ]

        self.clustering_algorithms = [
            "fuzzy c-means",
            "k-means",
            "agglomerative",
        ]

        self.dim_reduction_methods = [
            "None",
            "PCA",
            "UMAP"
        ]

        self.embedding_models = [
            "nomic-embed-text",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            # Agregar más modelos aquí
        ]

        self.llm_models = [
            "deepseek-ai/DeepSeek-V3-0324",
            "gpt-3.5-turbo",
            "gpt-4",
            # Agregar más modelos aquí
        ]

        self.llm_providers = [
            "fireworks-ai",
            "novita",
            "",
            # Agregar más proveedores aquí
        ]

    def _create_widgets(self):
        """Create all the Interface´s widgets"""

        # Frame principal con dos columnas
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Columna izquierda - Controles
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(side="left", fill="both", padx=(0, 5), pady=0)

        # Columna derecha - Consola
        console_frame = ctk.CTkFrame(main_frame)
        console_frame.pack(side="right", fill="both", expand=True, padx=(5, 0), pady=0)

        self._create_controls_section(controls_frame)
        self._create_console_section(console_frame)

    def _create_controls_section(self, parent):
        """Create control section."""

        # Título
        title_label = ctk.CTkLabel(parent, text="Requirements Generator",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(10, 2))

        # Scrollable frame para los controles
        scroll_frame = ctk.CTkScrollableFrame(parent, width=600, height=780)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 2))

        # Sección: Archivos de entrada
        self._create_file_section(scroll_frame)

        # Sección: Descripción de la aplicación
        self._create_app_description_section(scroll_frame)

        # Sección: Parámetros de filtrado
        self._create_filtering_section(scroll_frame)

        # Sección: Parámetros de clustering
        self._create_clustering_section(scroll_frame)

        # Sección: Parámetros de generación
        self._create_generation_section(scroll_frame)

    def _create_section_header(self, parent, title):
        """Create a section header."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(10, 5))

        label = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=5)

        return frame

    def _create_file_section(self, parent):
        """Create a file selection section."""
        self._create_section_header(parent, "📁 Files")

        # CSV Path
        csv_frame = ctk.CTkFrame(parent)
        csv_frame.pack(fill="x", pady=2)

        ctk.CTkLabel(csv_frame, text="Reviews (.csv):", width=120).pack(side="left", padx=(10, 5), pady=5)
        csv_entry = ctk.CTkEntry(csv_frame, textvariable=self.csv_path, width=200)
        csv_entry.pack(side="left", padx=5, pady=5)
        ctk.CTkButton(csv_frame, text="📂", width=30, command=self._select_csv_file).pack(side="left", padx=5, pady=5)

        # Output Directory
        output_frame = ctk.CTkFrame(parent)
        output_frame.pack(fill="x", pady=2)

        ctk.CTkLabel(output_frame, text="Output directory:", width=120).pack(side="left", padx=(10, 5), pady=5)
        output_entry = ctk.CTkEntry(output_frame, textvariable=self.output_directory, width=200)
        output_entry.pack(side="left", padx=5, pady=5)
        ctk.CTkButton(output_frame, text="📂", width=30, command=self._select_output_dir).pack(side="left", padx=5,
                                                                                              pady=5)

    def _create_app_description_section(self, parent):
        """Create app description section"""
        self._create_section_header(parent, "📝 Software description")

        desc_frame = ctk.CTkFrame(parent)
        desc_frame.pack(fill="x", pady=2)

        desc_textbox = ctk.CTkTextbox(desc_frame, height=60)
        desc_textbox.pack(fill="x", padx=10, pady=5)
        desc_textbox.insert("1.0", self.app_description.get())

        # Función para actualizar la variable cuando cambia el texto
        def update_description(event=None):
            self.app_description.set(desc_textbox.get("1.0", "end-1c"))

        desc_textbox.bind("<KeyRelease>", update_description)

    def _create_filtering_section(self, parent):
        """Create filtering parameters section"""
        self._create_section_header(parent, "🔍 Reviews Filtering")

        filter_frame = ctk.CTkFrame(parent)
        filter_frame.pack(fill="x", pady=2)

        # Grid layout para parámetros
        ctk.CTkLabel(filter_frame, text="Classification Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(filter_frame, variable=self.classification_model,
                        values=self.classification_models, width=200).grid(row=0, column=1, padx=10, pady=5)

        ctk.CTkLabel(filter_frame, text="Knowledge vector type:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(filter_frame, variable=self.vector_type,
                        values=self.vector_types, width=200).grid(row=1, column=1, padx=10, pady=5)

    def _create_clustering_section(self, parent):
        """Create clustering parameters section."""
        self._create_section_header(parent, "🔗 Reviews Clustering")

        cluster_frame = ctk.CTkFrame(parent)
        cluster_frame.pack(fill="x", pady=2)

        # Grid layout para parámetros
        ctk.CTkLabel(cluster_frame, text="Clustering algorithm:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(cluster_frame, variable=self.clustering_algorithm,
                        values=self.clustering_algorithms, width=180).grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(cluster_frame, text="Reduction algorithm.:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ctk.CTkComboBox(cluster_frame, variable=self.dim_reduction,
                        values=self.dim_reduction_methods, width=120).grid(row=0, column=3, padx=5, pady=5)

        ctk.CTkLabel(cluster_frame, text="K min:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(cluster_frame, textvariable=self.k_min, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(cluster_frame, text="K max:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(cluster_frame, textvariable=self.k_max, width=60).grid(row=1, column=3, padx=5, pady=5, sticky="w")

    def _create_generation_section(self, parent):
        """Create generation parameters section."""
        self._create_section_header(parent, "⚡ Requirement Generation")

        gen_frame = ctk.CTkFrame(parent)
        gen_frame.pack(fill="x", pady=2)

        # Grid layout para parámetros
        ctk.CTkLabel(gen_frame, text="LLM model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(gen_frame, variable=self.llm_model,
                        values=self.llm_models, width=300).grid(row=0, column=1, padx=10, pady=5)

        ctk.CTkLabel(gen_frame, text="Provider (HF API):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(gen_frame, variable=self.llm_provider,
                        values=self.llm_providers, width=200).grid(row=1, column=1, padx=10, pady=5, sticky="w")

    def _create_control_buttons(self, parent):
        """Create control buttons."""
        button_frame = ctk.CTkFrame(parent)
        button_frame.pack(fill="x", padx=10, pady=10)

        self.start_button = ctk.CTkButton(button_frame, text="🚀 Start Generation",
                                          command=self._start_generation,
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          height=40)
        self.start_button.pack(side="left", padx=5, expand=True, fill="x")

        self.stop_button = ctk.CTkButton(button_frame, text="⏹️ Stop",
                                         command=self._stop_generation,
                                         state="disabled",
                                         fg_color="red", hover_color="darkred",
                                         height=40)
        self.stop_button.pack(side="left", padx=5, expand=True, fill="x")

        clear_button = ctk.CTkButton(button_frame, text="🗑️ Clear Console",
                                     command=self._clear_console,
                                     height=40)
        clear_button.pack(side="left", padx=5, expand=True, fill="x")

    def _create_console_section(self, parent):
        """Create console section."""
        console_title = ctk.CTkLabel(parent, text="📟 Console",
                                     font=ctk.CTkFont(size=18, weight="bold"))
        console_title.pack(pady=(10, 5))

        # Frame para la consola
        console_container = ctk.CTkFrame(parent)
        console_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Textbox para la consola con scrollbar
        self.console_text = ctk.CTkTextbox(console_container,
                                           font=ctk.CTkFont(family="Consolas", size=11),
                                           wrap="word")
        self.console_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Estado inicial
        self._log_message("🎯 System ready. Set the parameters amd start generation.")

        # Botones de control
        self._create_control_buttons(console_container)

    def _setup_console_redirect(self):
        """Setup console redirect."""
        self.original_stdout = sys.stdout
        self.console_redirect = ConsoleRedirect(self.console_text, self.console_queue)

    def _start_console_update(self):
        """Start console periodic update."""
        self._update_console()
        self.root.after(100, self._start_console_update)

    def _update_console(self):
        """Update the console with new messages."""
        try:
            while True:
                message = self.console_queue.get_nowait()
                self.console_text.insert("end", message)
                self.console_text.see("end")
        except queue.Empty:
            pass

    def _log_message(self, message):
        """Add new message to console with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.console_queue.put(formatted_message)

    def _select_csv_file(self):
        """Select csv file."""
        file_path = filedialog.askopenfilename(
            title="Select csv file",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            self.csv_path.set(file_path)
            self._log_message(f"📁 Selected CSV: {os.path.basename(file_path)}")

    def _select_output_dir(self):
        """Select output directory."""
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            self.output_directory.set(dir_path)
            self._log_message(f"📁 Output directory: {dir_path}")

    def _validate_inputs(self):
        """Validate inputs before starting."""
        if not self.csv_path.get():
            messagebox.showerror("Error", "You must select a CSV file")
            return False

        if not os.path.exists(self.csv_path.get()):
            messagebox.showerror("Error", "The CSV file don´t exists")
            return False

        if not self.output_directory.get():
            messagebox.showerror("Error", "You must select a output directory")
            return False

        if self.k_min.get() >= self.k_max.get():
            messagebox.showerror("Error", "K min must be lower than K max")
            return False

        return True

    def _start_generation(self):
        """Start generation process."""
        if not self._validate_inputs():
            return

        if self.is_running:
            messagebox.showwarning("Warning", "There is already a process in execution")
            return

        # Cambiar estado de botones
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.is_running = True

        # Redirect stdout
        sys.stdout = self.console_redirect

        self._log_message("🚀 Starting Requirements Generation...")

        # Execute separated thread
        thread = threading.Thread(target=self._run_generation, daemon=True)
        thread.start()

    def _run_generation(self):
        """Execute the generation process in a separated thread."""
        try:
            # Update app description from textbox

            results = self.controller.generate_requirements_from_csv(
                csv_path=self.csv_path.get(),
                app_description=self.app_description.get(),
                output_directory=self.output_directory.get(),
                # Filtering parameters
                classification_model=self.classification_model.get(),
                vector_type=self.vector_type.get(),
                # Clustering parameters
                clustering_algorithm=self.clustering_algorithm.get(),
                dim_reduction=self.dim_reduction.get(),
                k_min=self.k_min.get(),
                k_max=self.k_max.get(),
                # Generation parameters
                llm_model=self.llm_model.get(),
                llm_provider=self.llm_provider.get()
            )

            self._log_message("✅ Generation completed successfully!")
            self._log_message(f"📊 Results saved at: {results['metadata']['run_directory']}")

        except Exception as e:
            self._log_message(f"❌ Error during generation: {str(e)}")

        finally:
            # Restaurar stdout
            sys.stdout = self.original_stdout

            # Restaurar estado de botones
            self.root.after(0, self._reset_buttons)

    def _stop_generation(self):
        """Stops the generation process."""
        self._log_message("⏹️ Stopping process...")
        self.is_running = False
        self._reset_buttons()

    def _reset_buttons(self):
        """Reset buttons state."""
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.is_running = False

    def _clear_console(self):
        """Clear the console."""
        self.console_text.delete("1.0", "end")
        self._log_message("🧹 Console cleared")

    def run(self):
        """Execute the app."""
        try:
            self.root.mainloop()
            self._log_message("🎮 Interface started successfully")
        except Exception as e:
            print(f"Error starting the app: {e}")
            messagebox.showerror("Fatal Error", f"Application could not start:\n{str(e)}")


def start():
    """Main Function."""
    try:
        app = RequirementsGeneratorGUI()
        app.run()
    except Exception as e:
        print(f"Error starting the app: {e}")
        messagebox.showerror("Fatal Error", f"Application could not start:\n{str(e)}")


if __name__ == "__main__":
    start()
