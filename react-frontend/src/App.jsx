import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import ContactPage from './pages/ContactPage';
import GithubPage from './pages/GithubPage';
import PredictionPage from './pages/PredictionPage';
import './App.css';

function App() {
  return (
    <>
      {/* <Navbar /> */}
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/contact" element={<ContactPage />} />
        <Route path="/github" element={<GithubPage />} />
        <Route path="/predict" element={<PredictionPage />} />
      </Routes>
    </>
  );
}

export default App;
