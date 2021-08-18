/**
 * ページ上でコンテストメニューを表示した時に発生するイベント
 */
document.addEventListener('contextmenu', function(event) {
	x = event.clientX * window.devicePixelRatio;
	y = event.clientY * window.devicePixelRatio;
}, true);

/**
 * Background Scriptからのメッセージを受け取るためのリスナー
 */
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
	if (request === 'getClickedPosition') {
		sendResponse({x: x, y: y});
		return true;
	}
});