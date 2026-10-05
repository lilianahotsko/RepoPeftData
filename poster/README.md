# Code2LoRA Conference Poster

High-quality conference poster for "Code2LoRA: Repository-Aware Adapter Generation via Composable Hypernetworks"

## Files

- **`code2lora_poster.html`** - Web-optimized version (1200px width)
  - Best for digital display and screen viewing
  - Responsive design with modern styling

- **`code2lora_poster_print.html`** - Print-optimized version (42" × 36")
  - Optimized for large-format poster printing
  - Uses `@page` CSS for proper print sizing
  - Includes print-safe color adjustments

## Viewing

Open either HTML file directly in a web browser.

## Exporting to PDF

### Option 1: Browser Print (Recommended)

1. Open `code2lora_poster_print.html` in Chrome or Firefox
2. Press `Ctrl+P` (or `Cmd+P` on Mac)
3. Set destination to "Save as PDF"
4. Set scale to "100%" or "Fit to page"
5. Enable "Background graphics" in More settings
6. Save

### Option 2: Command Line

```bash
# Using wkhtmltopdf
wkhtmltopdf --page-width 42in --page-height 36in \
  --enable-local-file-access \
  code2lora_poster_print.html poster.pdf

# Using Chrome headless
google-chrome --headless --print-to-pdf=poster.pdf \
  --no-pdf-header-footer \
  code2lora_poster_print.html
```

## Customization

The posters use CSS custom properties (variables) for easy theming. Edit the `:root` section to change:

- `--primary` / `--accent`: Main color scheme
- `--text-dark` / `--text-medium` / `--text-light`: Text colors
- `--bg-light` / `--bg-card`: Background colors

## Key Content

The poster highlights:

1. **Problem**: Context-based methods (RAG, ICL) have limitations for repository-level code understanding
2. **Solution**: Code2LoRA generates LoRA adapters from repository embeddings via hypernetworks
3. **Results**: 63.8% exact match on cross-repo eval (+18pp over pretrained)
4. **Efficiency**: Zero inference tokens, <10ms adapter generation

## Fonts

The poster uses Google Fonts:
- **Inter** - Primary text font
- **JetBrains Mono** - Code/numeric values

Ensure internet connectivity for font loading, or download and embed fonts locally for offline use.
