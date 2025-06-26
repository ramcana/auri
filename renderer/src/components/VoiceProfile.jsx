import React, { useState, useEffect } from 'react';
import './VoiceProfile.css';

const EDGE_TTS_VOICES = [
  { id: 'en-US-AriaNeural', name: 'Aria (US Female)', language: 'English (US)' },
  { id: 'en-US-GuyNeural', name: 'Guy (US Male)', language: 'English (US)' },
  { id: 'en-GB-SoniaNeural', name: 'Sonia (UK Female)', language: 'English (UK)' },
  { id: 'en-AU-NatashaNeural', name: 'Natasha (AU Female)', language: 'English (Australia)' },
  { id: 'en-IN-NeerjaNeural', name: 'Neerja (IN Female)', language: 'English (India)' }
];

const TTS_ENGINES = [
  { id: 'edge_tts', name: 'Microsoft Edge TTS', description: 'High-quality neural voices from Microsoft' },
  { id: 'dia', name: 'Nari Labs Dia', description: 'Advanced neural TTS with natural intonation (requires model)' },
  { id: 'gtts', name: 'Google TTS', description: 'Google Text-to-Speech service' },
  { id: 'pyttsx3', name: 'Local TTS', description: 'System voices (offline)' }
];

const VoiceProfile = ({ onSave, initialSettings }) => {
  const [settings, setSettings] = useState({
    engine: 'edge_tts',
    voice: 'en-US-AriaNeural',
    speed: 1.0,
    pitch: 1.0,
    ...initialSettings
  });
  
  const [previewText, setPreviewText] = useState("Hello! This is a preview of how the voice will sound.");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // Convert numeric values
    const numericValue = name === 'speed' || name === 'pitch' ? parseFloat(value) : value;
    
    setSettings({
      ...settings,
      [name]: numericValue
    });
  };
  
  const playPreview = async () => {
    try {
      setIsPlaying(true);
      
      // Call TTS service to generate preview
      const response = await fetch('http://localhost:8003/synthesize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: previewText,
          voice: settings.voice,
          speed: settings.speed,
          pitch: settings.pitch,
          engine: settings.engine
        })
      });
      
      if (!response.ok) {
        throw new Error(`TTS service error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Play the audio
      const audio = new Audio(`data:audio/${data.format};base64,${data.audio_data}`);
      audio.onended = () => setIsPlaying(false);
      audio.onerror = () => {
        console.error('Error playing audio preview');
        setIsPlaying(false);
      };
      
      await audio.play();
    } catch (error) {
      console.error('Error generating preview:', error);
      setIsPlaying(false);
    }
  };
  
  const handleSave = () => {
    setIsSaving(true);
    
    // Simulate API call to save settings
    setTimeout(() => {
      if (onSave) {
        onSave(settings);
      }
      setIsSaving(false);
    }, 500);
  };
  
  return (
    <div className="voice-profile-container">
      <h2>Voice Profile Settings</h2>
      
      <div className="form-group">
        <label htmlFor="engine">TTS Engine</label>
        <select 
          id="engine" 
          name="engine" 
          value={settings.engine} 
          onChange={handleChange}
          className="form-control"
        >
          {TTS_ENGINES.map(engine => (
            <option key={engine.id} value={engine.id}>
              {engine.name}
            </option>
          ))}
        </select>
        <small className="form-text">
          {TTS_ENGINES.find(e => e.id === settings.engine)?.description}
        </small>
      </div>
      
      <div className="form-group">
        <label htmlFor="voice">Voice</label>
        <select 
          id="voice" 
          name="voice" 
          value={settings.voice} 
          onChange={handleChange}
          className="form-control"
          disabled={settings.engine !== 'edge_tts'}
        >
          {EDGE_TTS_VOICES.map(voice => (
            <option key={voice.id} value={voice.id}>
              {voice.name} - {voice.language}
            </option>
          ))}
        </select>
        <small className="form-text">
          Voice selection is currently only available for Edge TTS
        </small>
      </div>
      
      <div className="form-row">
        <div className="form-group">
          <label htmlFor="speed">
            Speed: {settings.speed.toFixed(1)}x
          </label>
          <input 
            type="range" 
            id="speed" 
            name="speed" 
            min="0.5" 
            max="2.0" 
            step="0.1" 
            value={settings.speed} 
            onChange={handleChange}
            className="form-control-range"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="pitch">
            Pitch: {settings.pitch.toFixed(1)}
          </label>
          <input 
            type="range" 
            id="pitch" 
            name="pitch" 
            min="0.5" 
            max="2.0" 
            step="0.1" 
            value={settings.pitch} 
            onChange={handleChange}
            className="form-control-range"
          />
        </div>
      </div>
      
      <div className="preview-section">
        <h3>Voice Preview</h3>
        <textarea
          value={previewText}
          onChange={(e) => setPreviewText(e.target.value)}
          placeholder="Enter text to preview the voice"
          rows={3}
          className="form-control"
        />
        <button 
          onClick={playPreview} 
          disabled={isPlaying || !previewText.trim()}
          className="preview-button"
        >
          {isPlaying ? 'Playing...' : 'Play Preview'}
        </button>
      </div>
      
      <div className="actions">
        <button 
          onClick={handleSave} 
          disabled={isSaving}
          className="save-button"
        >
          {isSaving ? 'Saving...' : 'Save Voice Profile'}
        </button>
      </div>
    </div>
  );
};

export default VoiceProfile;
