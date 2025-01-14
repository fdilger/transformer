import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Dracula theme colors
const colors = {
  background: '#282a36',
  currentLine: '#44475a',
  foreground: '#f8f8f2',
  comment: '#6272a4',
  cyan: '#8be9fd',
  green: '#50fa7b',
  orange: '#ffb86c',
  pink: '#ff79c6',
  purple: '#bd93f9',
  red: '#ff5555',
  yellow: '#f1fa8c'
};

const TrainingMonitor = () => {
  const [trainingStats, setTrainingStats] = useState({
    loss: [{time: 0, value: 0}], // Initialize with a starting point
    accuracy: [{time: 0, value: 0}],
    learningRate: [{time: 0, value: 0}],
    epochProgress: 0,
    timeRemaining: '00:00:00',
    currentEpoch: 1,
    totalEpochs: 10,
    warnings: []
  });

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket Connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTrainingStats(prev => ({
        ...prev,
        loss: [...prev.loss, { time: prev.loss.length, value: data.loss }].filter(Boolean),
        accuracy: [...prev.accuracy, { time: prev.accuracy.length, value: data.accuracy }].filter(Boolean),
        learningRate: [...prev.learningRate, { time: prev.learningRate.length, value: data.learning_rate }].filter(Boolean),
        epochProgress: data.epoch_progress,
        timeRemaining: data.time_remaining,
        currentEpoch: data.current_epoch,
        totalEpochs: data.total_epochs,
        warnings: data.warnings
      }));
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected, attempting to reconnect...');
      setTimeout(() => {
        console.log('Attempting to reconnect...');
        connectWebSocket();
      }, 1000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    return ws;
  }, []);

  useEffect(() => {
    const ws = connectWebSocket();
    return () => ws.close();
  }, [connectWebSocket]);

  const ChartComponent = ({ data, title, dataKey, color, domain }) => (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-gray-100">{title}</h2>
      </div>
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart 
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={colors.currentLine} />
            <XAxis 
              dataKey="time" 
              label={{ value: 'Steps', position: 'bottom', fill: colors.foreground }} 
              tick={false}
              stroke={colors.foreground}
            />
            <YAxis 
              domain={domain || ['auto', 'auto']}
              tick={{ fill: colors.foreground }}
              stroke={colors.foreground}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: colors.background,
                border: `1px solid ${colors.currentLine}`,
                color: colors.foreground
              }}
            />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke={color} 
              dot={false}
              strokeWidth={2}
              isAnimationActive={false} // Disable animation for real-time updates
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  return (
    <div className="p-6 space-y-6 bg-gray-900 min-h-screen" style={{ width: '100vw' }}>
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-100">(˶˃ ᵕ ˂˶) .ᐟ.ᐟ</h1>
        <div className="flex items-center space-x-6 text-gray-100">
          <span>epoch: {trainingStats.currentEpoch} of {trainingStats.totalEpochs}</span>
        </div>
      </div>

      {/* Main metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartComponent 
          data={trainingStats.loss}
          title="loss"
          dataKey="value"
          color={colors.purple}
        />
        <ChartComponent 
          data={trainingStats.accuracy}
          title="val loss"
          dataKey="value"
          color={colors.green}
          domain={[0, 1]}
        />
        <ChartComponent 
          data={trainingStats.learningRate}
          title="lr"
          dataKey="value"
          color={colors.cyan}
        />

        {/* progress */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <div className="mb-4">
            <h2 className="text-xl font-semibold text-gray-100">Epoch Progress</h2>
          </div>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full bg-gray-700 text-gray-100">
                  Progress
                </span>
              </div>
              <div className="text-right">
                <span className="text-xs font-semibold inline-block text-gray-100">
                  {trainingStats.epochProgress}%
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-3 mb-4 text-xs flex rounded bg-gray-700">
              <div 
                style={{ width: `${trainingStats.epochProgress}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-gray-100 justify-center bg-purple-500 transition-all duration-500"
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Warnings and Alerts */}
      {trainingStats.warnings.length > 0 && (
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <div className="flex items-center mb-2">
            <h2 className="text-xl font-semibold text-orange-400">Warnings</h2>
          </div>
          <ul className="list-disc pl-5">
            {trainingStats.warnings.map((warning, index) => (
              <li key={index} className="text-orange-400">{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default TrainingMonitor;