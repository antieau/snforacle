document.addEventListener('DOMContentLoaded', function () {
  var el = document.getElementById('snf-demo');
  if (!el) return;
  AsciinemaPlayer.create(
    'assets/snf_demo.cast',
    el,
    { autoPlay: true, loop: true, speed: 1.5, theme: 'solarized-dark',
      terminalFontSize: 'small', fit: 'width' }
  );
});
