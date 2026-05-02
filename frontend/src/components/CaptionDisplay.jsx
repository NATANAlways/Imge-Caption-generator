export default function CaptionDisplay({ result, loading, error }) {
  if (loading) return (
    <p style={{ color: '#666', fontStyle: 'italic' }}>Generating caption...</p>
  )

  if (error) return (
    <p style={{ color: 'red' }}>Error: {error}</p>
  )

  if (!result) return null

  return (
    <div style={{
      background: 'white',
      padding: '20px',
      borderRadius: '10px',
      marginTop: '20px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
    }}>
      <p style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '10px' }}>
        "{result.caption}"
      </p>
      <p style={{ color: '#666', fontSize: '13px' }}>
        Model: {result.model.toUpperCase()} &nbsp;|&nbsp; Time: {result.time_ms} ms
      </p>
    </div>
  )
}
