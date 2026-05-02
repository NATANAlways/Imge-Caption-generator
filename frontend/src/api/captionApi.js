import axios from 'axios'

const BASE_URL = 'http://localhost:8000/api'

export async function generateCaption(imageFile, modelName) {
  const formData = new FormData()
  formData.append('file', imageFile)

  const response = await axios.post(
    `${BASE_URL}/caption?model_name=${modelName}`,
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  )

  return response.data  // { caption, model, time_ms }
}
