const copyButton = document.getElementById('copy-citation');
const copyStatus = document.getElementById('copy-status');
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex');
  try {
    await navigator.clipboard.writeText(citation.textContent.trim());
    copyButton.textContent = 'Copied!';
    copyStatus.textContent = 'Citation copied to clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 2000);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    copyButton.textContent = 'Select & copy';
  }
});
