import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'

function PrivateRoute({ children }) {
  return localStorage.getItem('sg_token') ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/*" element={<PrivateRoute><ChatPage /></PrivateRoute>} />
      </Routes>
    </BrowserRouter>
  )
}
