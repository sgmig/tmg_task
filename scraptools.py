#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This files contains functions and classes used for scrapping. The first versions of these functions 
# were developped and are explained and tested in the notebook scrapping_trials.ipynb

import os
import time
import random

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from multiprocessing import Pool


# In[ ]:


def get_header():
    """
    Returns a random header dictionary to be passed to requests. 
    
    It'd be better to read the headers from an external file.
    """

    # Headers for user agent rotation:
    # Full headers obtained from hhttpbin.org
    
    
    # Firefox 84 Ubuntu
    h1 =  {
        "Accept": 	"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding":	"gzip, deflate, br",
        "Accept-Language":	"en-US,en;q=0.5",
        "Connection":	"keep-alive",
        "Host":	"httpbin.org",
        "TE":	"Trailers",
        "Upgrade-Insecure-Requests":	"1",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0"
      }

    #Firefox 84 Windows 10

    h2 = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8", 
        "Accept-Encoding": "gzip, deflate, br", 
        "Accept-Language": "en-GB,en;q=0.5", 
        "Host": "httpbin.org", 
        "Upgrade-Insecure-Requests": "1", 
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0"
       }

    # Chrome 87 Ubuntu

    h3 = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", 
        "Accept-Encoding": "gzip, deflate, br", 
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,es;q=0.7", 
        "Host": "httpbin.org", 
        "Sec-Fetch-Dest": "document", 
        "Sec-Fetch-Mode": "navigate", 
        "Sec-Fetch-Site": "none", 
        "Sec-Fetch-User": "?1", 
        "Upgrade-Insecure-Requests": "1", 
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36", 
      }

    #Chrome 87 Windows 10

    h4 = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", 
        "Accept-Encoding": "gzip, deflate", 
        "Accept-Language": "es-419,es;q=0.9,fr;q=0.8,en;q=0.7", 
        "Host": "httpbin.org", 
        "Upgrade-Insecure-Requests": "1", 
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
      }

    # Microsoft Edge 87 Windows 10

    h5 = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", 
        "Accept-Encoding": "gzip, deflate, br", 
        "Accept-Language": "en-US,en;q=0.9", 
        "Host": "httpbin.org", 
        "Sec-Fetch-Dest": "document", 
        "Sec-Fetch-Mode": "navigate", 
        "Sec-Fetch-Site": "none", 
        "Sec-Fetch-User": "?1", 
        "Upgrade-Insecure-Requests": "1", 
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75"
      }

    headers_list = [h1, h2, h3, h4, h5]
    
    return random.choice(headers_list)


# In[ ]:


def get_links(soup):
    """ Get get all the links for the given 'a' html tag from a parsed page.  
    soup: an html page parsed with beautifulsoup.
    """
    
    links = [] # list to store the links found
    
    for link in soup.find_all('a', href=True):
        links.append(link['href'])
           
    # avoiding repetitions
    links = list(set(links))
        
        
    return links


# In[ ]:


def filter_links(home, links_list):
    """
    Takes a home address and a list of links, and filters out links to external sites
    and to some common file types.
    home: string. The URL of the home page.
    links_list: list of strings with the links found on the page, as produced by get_links.
    """
    
    domain = urlparse(home).netloc # domain to to check for external links.
    
    # path to include before an internal link. Remove final '/' if present.
    path = home[:-1] if home.endswith('/') else home 

    unwanted_starts = ('javascript:', 'mailto:', 'tel:', '#', '..', '../', 'stream') 
    
    unwanted_endings = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.exe', '.js',
                        '.zip', '.tar', '.gz', '.7z', '.rar'
                       )
    
    filtered_links = list(filter(lambda link: not (link.lower().startswith(unwanted_starts) or 
                                                   link.lower().endswith(unwanted_endings)),links_list
                                )
                         )
    
    # get internal links that don't have the full URL
    internal_links = [link for link in filtered_links if not link.startswith('http') ]

    # Ensure starting '/'  
    for j, intlink in enumerate(internal_links):
        if not intlink.startswith('/'):
            internal_links[j]='/'+intlink
            
    internal_links = [path + intlink for intlink in internal_links]
    
    # removing external links
    filtered_links = list(filter(lambda link: (link.lower().startswith('http') and
                                                domain in link.lower()), filtered_links
                                )
                         )
    
    # include internal links
    filtered_links.extend(internal_links)
    
    # keeping disntinct elements only
    
    filtered_links = list(set(filtered_links))
    
    # remove home url if present.    
    try:
        filtered_links.remove(path)
    except(ValueError):
        pass
    try:
        filtered_links.remove(path+'/')
    except(ValueError):
        pass
        
    return filtered_links


# In[ ]:


class UnwantedContentError(Exception):
    """ Raised if the content of the site was not text """
    
    def __init__(self, content_type):
        self.content_type = content_type
    
    def message(self):
        
        msg = f'Unwanted content type:  {self.content_type} .'
        
        return msg


# In[2]:


def scrape_main(main_url):
    
    """
    Takes the URL of the main site, scrapes the text and the links. 
    site_url: string. url of the desired site.
    """
    
    random_header = get_header()
    
    page = requests.get(main_url, {'header': random_header}, timeout=(3, 5))
    
    # Verify we didn't get and invalid response.
    page.raise_for_status()
    
    # Verify content type is text.
    if not page.headers['Content-Type'].startswith('text'):
        raise UnwantedContentError(page.headers['Content-Type'])
    else:    
        soup = BeautifulSoup(page.content, 'html.parser')


        page_text = soup.get_text(separator = '\n', strip=True) 

        page_links = get_links(soup)
        page_links = filter_links(main_url, page_links)

        return page_text, page_links   


# In[ ]:


def scrape_links(link_url):
    
    """
    Takes the URL from one of the link, scrapesa and returns the text. 
    link_url: string. url of the desired site.
    """
    
    random_header = get_header()
    
      
    page = requests.get(link_url, params = {'header': random_header}, timeout=(5, 5))
    
    # Verify we didn't get and invalid response. 
    page.raise_for_status()    
    
    # Verify content type is text.
    if not page.headers['Content-Type'].startswith('text'):
        raise UnwantedContentError(page.headers['Content-Type'])
    else:
        soup = BeautifulSoup(page.content, 'html.parser')

        page_text = soup.get_text(separator = '\n', strip=True) 

        return page_text


# In[ ]:


def scrape_links_try(mylink_url):
    """
    Scrape links and catch exceptions. This is a wrapper function intended to be passed
    to a multiplrocessing.Pool object.
    """
    
    try:
        scrapping_result = scrape_links(mylink_url)
    except requests.exceptions.RequestException as err:
        scrapping_result = [mylink_url, err]
    except UnwantedContentError as err:
        scrapping_result = [mylink_url, err.message()]
        print(f'WARNING: Content error in \n {mylink_url} \n {scrapping_result[1]} ')
        
    return scrapping_result


# In[ ]:


def scrape_full(sites_list, contents_dir = None, logs_dir = None, max_links = 20, n_process = None ):
    
    """
    Function to scrape all sites and links on the main page, extracting the text of each
    site to a text file.
    Creates a log file for the scrapping as a whole. Creates a link scrapping report for sites
    where more than half the links failed. 
    Returns a report_dict with the main information of the scrapping.
    
    sites_list: list. A list of sites urls in string format.
    contents_dir: string. Directory to store the text files with the contents. If None, a 
                  default directory is used. 
    logs_dir: string. Directory to store the text files with the scrapping logs. If None, a 
                  default directory is used. 
    max_links: int. Maximum number of links to take from each site. Note: Some of the links
               may not have scrappable contents. 
    n_process: int. Number of processes to be passed to Pool for performing the parallell scraping 
               of each site's links. If None, os.cpu_count() is used. 
               WARNING: Setting n_process greater thant os.cpu_count() may cause crash.
               A check for this will be included soon. 
    """
    
    # Setting default directories if none are provided.
    if contents_dir is None:
        contents_dir = './site_contents/'
    if logs_dir is None:        
        logs_dir = './logs/'

    # Make contents and logs directories if they don't exist.
    os.makedirs(contents_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Keep track of the result of each request. 
    failed_requests = []
    succes_requests = []

    # Time string to identify the corresponding log. 
    timestr = time.strftime("%Y%m%d_%H%M", time.localtime())
    
    start_time = time.time() # save for timing

    for site in sites_list:

        # Get links and text from the main site. Catch request errors and UnwantedContents. 

        try:
            text, links_list = scrape_main(site)
            succes_requests.append(site)
        except requests.exceptions.RequestException as err:
            failed_requests.append((site,str(err)))
            print(f'Error in site: {site}')
        except UnwantedContentError as err:
            failed_requests.append((site,err.message()))
        else:   

            # Create site contents file and write text of the main site
            domain = urlparse(site).netloc
            file_name = contents_dir + domain +'.txt'

            with open(file_name, 'w') as f:
                f.write(text)

            print(f'Text from main page {domain} written to {file_name}')

            start_time_p = time.time()
            print(f'Scrapping links from {domain}')

            links_list = links_list[:max_links] # Truncate link list.
            
            # Use multiprocessing pool for scrapping links.
            with Pool(n_process) as p:
                link_scrap_results = p.map(scrape_links_try, links_list)

            duration_p = time.time() - start_time_p

            print(f'{len(links_list)} links scrapped in {duration_p:.2f} seconds. ')
            
            
            # Separte link's text from failed links.
            text = list(filter(lambda result: type(result) is str,link_scrap_results))
            link_errors = list(filter(lambda result: type(result) is list,link_scrap_results))
            link_errors = [[site, str(err)] for site, err in link_errors]

            print(f'{len(link_errors)} links failed.')

            # Write link text to site file.
            with open(file_name, 'a') as f:
                f.write('\n'.join(text))

            # Log link errors only if more than half fail.
            if len(link_errors) >= len(link_scrap_results)/2 : 

                link_errors = ['\n'.join(error) for error in link_errors]

                with open(logs_dir+timestr+ '_link_report_' + domain+'.txt', 'w') as link_log:

                        link_log.write('\n\n'.join(link_errors))
                        
    duration = time.time()-start_time
    
    report_dict = {'sites': sites_list, 
                   'succesful': succes_requests,
                   'failed': failed_requests,
                   'time_s': duration,
                   'contents' : contents_dir,
                   'logs': logs_dir,
                   'report_name': logs_dir+timestr+'_scrapping_repport'
                  }

    report_str = (f'{len(sites_list) } requested. \n' 
                  + f'Scrapping took {duration/60:.2f} min ({duration:.2f} s) \n' 
                  + f'{len(succes_requests) } SUCCESFUL. \n'
                  + f'{len(failed_requests) } FAILURES. \n'
                  + '='*20 +'\n FAILED SITES \n' + '='*20 + '\n\n')


    with open(report_dict['report_name'], 'w') as scrapping_log:

        scrapping_log.write(report_str)
        scrapping_log.writelines(['\n'.join(failure)+'\n\n' for failure in failed_requests])

        scrapping_log.write('\n\n SUCCESFUL REQUESTS \n\n')
        scrapping_log.write('\n'.join(succes_requests))
        
    return report_dict

