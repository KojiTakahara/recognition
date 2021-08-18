'use strict';

chrome.runtime.onInstalled.addListener(() => {
	const parent = chrome.contextMenus.create({
		type: 'normal',
		id: 'parent',
		title: '画像を検索するよ！',
		contexts: ['all']
	});
});
chrome.contextMenus.onClicked.addListener(async (item, tab) => {
	chrome.tabs.sendMessage(tab.id, 'getClickedPosition', function(res) {
		chrome.tabs.query({active: true, currentWindow: true}, tabs => {
			if (0 < tabs.length) {
				const activeTab = tabs[0];
				chrome.tabs.captureVisibleTab(activeTab.windowId, {format: 'png'}, image => {
					let formData = new FormData();
					formData.append('file', image);
					formData.append('x', res.x);
					formData.append('y', res.y);
					let promise = fetchAPI(formData);
					promise.then((response) => response.text()).then((text) => {
						const obj = JSON.parse(text);
						if (obj.urls.length === 0) {
							alert('識別に失敗');
							return;
						}
						const list = obj.urls.filter((url) => {
							return (url.match(/torekakaku/)
								|| url.match(/static.wikia.nocookie/)
								|| url.match(/ka-nabell/)
								|| url.match(/gachi-matome/)
								|| url.match(/cardbox/));
						});
						if (list.length === 0) {
							list.push(obj.urls[0]);
						}
						const options = {
							url: list[0],
							type: 'popup',
							width: 300,
							height: 400
						};
						chrome.windows.create(options);
					}).catch((error) => {
						console.log(error);
						alert('識別に失敗');
					});
				})
            }
		})
	});
});
function fetchAPI(formData) {
	const url = 'http://127.0.0.1:8000/post';
	const options = {
		method: 'POST',
		body: formData,
		headers: {
			Accept: 'application/json',
			'Content-Type': 'multipart/form-data',
			enctype:'multipart/form-data',
		},
	};
	delete options.headers['Content-Type'];
	return fetch(url, options);
}

