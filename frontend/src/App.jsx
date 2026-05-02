import { useState } from 'react'
import ImageUploader    from './components/ImageUploader'
import ModelSelector    from './components/ModelSelector'
import CaptionDisplay   from './components/CaptionDisplay'
import { generateCaption } from './api/captionApi'

export default function App() {
  const [imageFile,  setImageFile]  = useState(null)
  const [modelName,  setModelName]  = useState('v2')
  const [result,     setResult]     = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)

  async function handleGenerate() {
    if (!imageFile) {
      alert('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await generateCaption(imageFile, modelName)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '600px', margin: '40px auto', padding: '24px',
                  background: 'white', borderRadius: '12px',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>

      <h1 style={{ marginBottom: '24px', color: '#333' }}>
        Image Caption Generator
      </h1>

      <ModelSelector selected={modelName} onChange={setModelName} />

      <ImageUploader onImageSelect={setImageFile} />

      <button
        onClick={handleGenerate}
        disabled={loading}
        style={{
          padding: '10px 28px', fontSize: '15px',
          background: loading ? '#aaa' : '#4f46e5',
          color: 'white', border: 'none',
          borderRadius: '8px', cursor: loading ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? 'Generating...' : 'Generate Caption'}
      </button>

      <CaptionDisplay result={result} loading={loading} error={error} />
    </div>
  )
}
