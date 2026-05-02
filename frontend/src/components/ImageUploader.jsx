import { useState } from 'react'

export default function ImageUploader({ onImageSelect }) {
  const [preview, setPreview] = useState(null)

  function handleFileChange(e) {
    const file = e.target.files[0]
    if (!file) return

    // Show preview immediately before sending to backend
    setPreview(URL.createObjectURL(file))

    // Send the actual file object up to App.jsx
    onImageSelect(file)
  }

  return (
    <div style={{ marginBottom: '20px' }}>
      <input
        type="file"
        accept="image/jpeg, image/png, image/webp"
        onChange={handleFileChange}
        style={{ marginBottom: '12px', display: 'block' }}
      />
      {preview && (
        <img
          src={preview}
          alt="preview"
          style={{ maxWidth: '400px', borderRadius: '8px', border: '1px solid #ddd' }}
        />
      )}
    </div>
  )
}