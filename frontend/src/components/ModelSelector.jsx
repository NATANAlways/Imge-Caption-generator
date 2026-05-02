export default function ModelSelector({ selected, onChange }) {
  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{ fontWeight: 'bold', marginRight: '10px' }}>
        Select Model:
      </label>
      <select
        value={selected}
        onChange={e => onChange(e.target.value)}
        style={{ padding: '8px 12px', borderRadius: '6px', fontSize: '14px' }}
      >
        <option value="v1">V1 — Custom CNN + LSTM</option>
        <option value="v2">V2 — ResNet50 + Attention</option>
      </select>
    </div>
  )
}
