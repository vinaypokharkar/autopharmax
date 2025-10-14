import React from 'react';
import './WhyUs.css';

const WhyUs = () => {
  return (
    <div className="why-us">
      <h2>Why Us?</h2>
      <div className="features">
        <div className="feature">
          <div className="feature-number">1</div>
          <div className="feature-content">
            <h3>Unmatched Accuracy</h3>
            <p>Our fine-tuned XGBoost model achieves a 99.22% R2 Score, ensuring predictions that closely mirror real-world lab outcomes.</p>
          </div>
        </div>
        <div className="feature">
          <div className="feature-number">2</div>
          <div className="feature-content">
            <h3>High Correlation</h3>
            <p>We demonstrate a 99.62% Pearson correlation with actual experimental values, validating our model's predictive power.</p>
          </div>
        </div>
        <div className="feature">
          <div className="feature-number">3</div>
          <div className="feature-content">
            <h3>Impressive Low Error</h3>
            <p>With a Root Mean Square Error (RMSE) of only 0.2512, our predictions are precise, reliable, and ready for critical research applications.</p>
          </div>
        </div>
        <div className="feature">
          <div className="feature-number">4</div>
          <div className="feature-content">
            <h3>Production Ready & Scalable</h3>
            <p>AutoPharmaX is a fully deployed application, built on a robust pipeline ready to integrate into real-world research workflows and scale with your demands.</p>
          </div>
        </div>
        <div className="feature">
          <div className="feature-number">5</div>
          <div className="feature-content">
            <h3>End-to-End Integrity</h3>
            <p>Our predictions are powered by a comprehensive data pipeline, from data merging and feature engineering to final deployment, ensuring quality, consistency, and traceability.</p>
          </div>
        </div>
        <div className="feature">
          <div className="feature-number">6</div>
          <div className="feature-content">
            <h3>Best-in-Class Technology</h3>
            <p>To provide the best technology, we rigorously tested multiple models. Our tuned XGBoost was chosen for its demonstrably superior performance.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WhyUs;
