import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
      </div>
      <div className="navbar-center">
        <Link to="/" className="navbar-brand">AutoPharmaX</Link>
      </div>
      <div className="navbar-right">
        <Link to="/github">Github</Link>
        <Link to="/contact">Contact</Link>
      </div>
    </nav>
  );
};

export default Navbar;
