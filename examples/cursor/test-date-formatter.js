// Test file for the date formatter function
const { formatDate, formatDatePreset, dateFormats } = require('./dateFormatter.js');

console.log('=== Date Formatter Test ===\n');

// Test with current date
const now = new Date();
console.log('Current date:', now);
console.log('ISO format:', formatDate(now));
console.log('US format:', formatDate(now, 'MM/DD/YYYY'));
console.log('EU format:', formatDate(now, 'DD/MM/YYYY'));
console.log('DateTime format:', formatDate(now, 'YYYY-MM-DD HH:mm:ss'));

console.log('\n=== Using Presets ===');
console.log('ISO preset:', formatDatePreset(now, 'ISO'));
console.log('US preset:', formatDatePreset(now, 'US'));
console.log('EU preset:', formatDatePreset(now, 'EU'));
console.log('DateTime preset:', formatDatePreset(now, 'DATETIME'));

console.log('\n=== Test with String Date ===');
const dateString = '2024-01-15T10:30:00Z';
console.log('String date:', dateString);
console.log('Formatted:', formatDate(dateString, 'DD/MM/YYYY HH:mm'));

console.log('\n=== Available Formats ===');
console.log('Available presets:', Object.keys(dateFormats));
