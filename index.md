\## Papers



<ul>

{% for post in site.posts %}

&nbsp; <li>

&nbsp;   <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>

&nbsp;   <small>({{ post.date | date: "%Y-%m-%d" }})</small>

&nbsp; </li>

{% endfor %}

</ul>



